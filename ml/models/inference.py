"""
Model Inference and Scoring Service.

Handles both supervised (CNN / EfficientNet) and unsupervised (normal-reference
autoencoder / statistical distance) anomaly evaluation.
Implements graceful MODEL_UNAVAILABLE handling when model weights are not found.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch

from ml.models.cnn import MachineAcousticCNN, SpectrogramAutoencoder
from ml.models.efficientnet import AcousticEfficientNet
from ml.preprocessing.audio import AudioPreprocessor
from ml.preprocessing.spectrogram import MelSpectrogramExtractor, SpectrogramConfig


class MachineAnomalyInference:
    """
    ML Inference Engine.
    Handles audio feature extraction, spectrogram synthesis, and model scoring.
    """

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.model_dir = Path(model_dir or "ml/weights")
        self.preprocessor = AudioPreprocessor(target_sample_rate=16000)
        self.spec_extractor = MelSpectrogramExtractor()

        self.model_type: Optional[str] = None
        self.model: Optional[torch.nn.Module] = None
        self.config: Dict[str, Any] = {}
        self.class_names: list = ["normal", "anomaly"]
        self.threshold: float = 0.50

        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load trained model artifacts from model_dir."""
        config_path = self.model_dir / "config.json"
        weights_path = self.model_dir / "model.pt"

        if not (config_path.exists() and weights_path.exists()):
            # Model artifacts are not yet trained
            self.model = None
            return

        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)

            self.model_type = self.config.get("model_type", "cnn").lower()
            self.threshold = float(self.config.get("threshold", 0.50))
            is_unsupervised = self.config.get("is_unsupervised", False)

            if is_unsupervised:
                self.model = SpectrogramAutoencoder()
            elif self.model_type == "efficientnet":
                self.model = AcousticEfficientNet(num_classes=2, pretrained=False)
            else:
                self.model = MachineAcousticCNN(num_classes=2)

            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()

            class_file = self.model_dir / "class_names.json"
            if class_file.exists():
                with open(class_file, "r") as f:
                    self.class_names = json.load(f)
        except Exception:
            self.model = None

    @property
    def is_available(self) -> bool:
        """Returns True if a trained model is loaded and ready for inference."""
        return self.model is not None

    def predict_file(
        self,
        audio_path: Union[str, Path],
        channel_index: Optional[int] = None,
        save_spectrogram_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes anomaly inference on an audio file.
        Returns explainable diagnosis or MODEL_UNAVAILABLE.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract audio and Mel spectrogram
        audio = self.preprocessor.process(str(path), channel_index=channel_index)
        mel_spec = self.spec_extractor.compute(audio)

        spectrogram_image_path = None
        if save_spectrogram_to:
            spectrogram_image_path = self.spec_extractor.save_image(mel_spec, save_spectrogram_to)

        if not self.is_available:
            return {
                "status": "MODEL_UNAVAILABLE",
                "result": "MODEL_UNAVAILABLE",
                "anomaly_score": None,
                "confidence": None,
                "model_version": "None",
                "spectrogram_path": spectrogram_image_path,
                "detail": "No trained model weights found in ml/weights/. Run training pipeline first.",
            }

        # Convert to tensor: (1, 1, n_mels, n_frames)
        tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            if isinstance(self.model, SpectrogramAutoencoder):
                # Unsupervised reconstruction error
                recon_error = self.model.compute_anomaly_score(tensor)
                # Map MSE error to [0, 1] relative to calibrated threshold
                normalized_score = float(np.clip(recon_error / (self.threshold * 2.0), 0.0, 1.0))
                is_anomaly = recon_error >= self.threshold
                result = "ANOMALY" if is_anomaly else "NORMAL"
                confidence = float(np.clip(abs(recon_error - self.threshold) / self.threshold, 0.5, 0.99))

                return {
                    "status": "SUCCESS",
                    "result": result,
                    "anomaly_score": round(recon_error, 5),
                    "confidence": round(confidence, 4),
                    "model_version": f"Autoencoder-{self.config.get('version', 'v1')}",
                    "spectrogram_path": spectrogram_image_path,
                    "is_unsupervised": True,
                }
            else:
                # Supervised classifier
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_idx = int(np.argmax(probs))
                anomaly_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                confidence = float(probs[pred_idx])

                result = "ANOMALY" if anomaly_prob >= self.threshold else "NORMAL"

                return {
                    "status": "SUCCESS",
                    "result": result,
                    "anomaly_score": round(anomaly_prob, 4),
                    "confidence": round(confidence, 4),
                    "model_version": f"{self.model_type.upper()}-{self.config.get('version', 'v1')}",
                    "spectrogram_path": spectrogram_image_path,
                    "is_unsupervised": False,
                }
