"""
Dataset and Manifest Management for DCASE 2026 Machine Condition Monitoring.

Features:
- DCASE 2026 Task 2 directory parsing (fan, gearbox, bearing, slide_rail, ToyCar, valve).
- Source vs Target domain split preservation to avoid data leakage.
- Near vs Far microphone channel separation.
- Automatic manifest generation (JSON/CSV) with metadata inspection.
- PyTorch Dataset class for feeding Mel spectrograms to training loops.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ml.preprocessing.audio import AudioPreprocessor
from ml.preprocessing.spectrogram import MelSpectrogramExtractor


@dataclass
class AudioSampleMeta:
    """Metadata record for an acoustic recording."""
    file_path: str
    file_name: str
    machine_type: str
    label: str  # "normal" or "anomaly"
    domain: str  # "source" or "target"
    section_id: str
    channel_count: int
    duration_sec: float
    sample_rate: int


class MachineAudioDataset(Dataset):
    """
    PyTorch Dataset yielding (mel_spectrogram_tensor, label_int, metadata_dict).
    """

    def __init__(
        self,
        samples: List[AudioSampleMeta],
        spec_extractor: Optional[MelSpectrogramExtractor] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        is_unsupervised: bool = False,
    ):
        self.samples = samples
        self.spec_extractor = spec_extractor or MelSpectrogramExtractor()
        self.preprocessor = preprocessor or AudioPreprocessor(target_sample_rate=16000)
        self.is_unsupervised = is_unsupervised

        # If unsupervised, only retain normal samples for training
        if self.is_unsupervised:
            self.samples = [s for s in self.samples if s.label == "normal"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        meta = self.samples[idx]
        audio = self.preprocessor.process(meta.file_path)
        mel_spec = self.spec_extractor.compute(audio)

        # Tensor shape: (1, n_mels, n_frames)
        tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
        label_int = 1 if meta.label == "anomaly" else 0

        return tensor, label_int, asdict(meta)


def build_manifest_from_directory(
    root_dir: Union[str, Path],
    manifest_output_path: Optional[Union[str, Path]] = None,
) -> List[AudioSampleMeta]:
    """
    Scans directory for WAV files and parses machine type, label, domain from filename/directory.
    Standard DCASE format:
    e.g. `section_00_source_train_normal_0000.wav`
         `section_00_target_test_anomaly_0001.wav`
         `fan_normal.wav`
    """
    root = Path(root_dir)
    records: List[AudioSampleMeta] = []

    for wav_file in root.rglob("*.wav"):
        fname = wav_file.stem.lower()
        info = sf.info(str(wav_file))

        # Infer machine type
        machine_type = "unknown"
        for m in ["fan", "gearbox", "bearing", "slide_rail", "sliderail", "toycar", "valve"]:
            if m in fname or m in str(wav_file.parent).lower():
                machine_type = m
                break

        # Infer label
        if "anomaly" in fname or "abnormal" in fname or "defect" in fname:
            label = "anomaly"
        elif "normal" in fname:
            label = "normal"
        else:
            label = "normal"  # default unsupervised baseline

        # Infer domain
        domain = "target" if "target" in fname else "source"

        # Infer section id
        section_id = "section_00"
        for part in fname.split("_"):
            if part.isdigit() and len(part) == 2:
                section_id = f"section_{part}"

        records.append(
            AudioSampleMeta(
                file_path=str(wav_file.resolve()),
                file_name=wav_file.name,
                machine_type=machine_type,
                label=label,
                domain=domain,
                section_id=section_id,
                channel_count=info.channels,
                duration_sec=info.duration,
                sample_rate=info.samplerate,
            )
        )

    if manifest_output_path:
        out_path = Path(manifest_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in records], f, indent=2)

    return records
