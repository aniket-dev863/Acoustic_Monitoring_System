"""
FastAPI Microservice for Acoustic Anomaly Machine Learning and Gating.

Endpoints:
- GET  /health
- GET  /model-info
- POST /predict
- POST /extract-spectrogram
- POST /gate-and-predict
"""

from pathlib import Path
import shutil
import tempfile
import time
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dsp.adaptive_gate import AdaptiveMultiBandResourceGate
from dsp.baseline_gates import (
    BaselineAdaptiveSNRGate,
    BaselineFixedAmplitudeGate,
    BaselineFixedBandGate,
)
from ml.models.inference import MachineAnomalyInference
from ml.preprocessing.spectrogram import MelSpectrogramExtractor

app = FastAPI(
    title="Acoustic Monitoring ML & Gate Service",
    version="1.0.0",
    description="Microservice providing edge acoustic gating and deep anomaly classification",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = MachineAnomalyInference()
spec_extractor = MelSpectrogramExtractor()


class PredictRequest(BaseModel):
    audio_path: str
    channel_index: Optional[int] = None


@app.get("/health")
def health_check():
    """Health check reporting service readiness and model status."""
    return {
        "status": "healthy",
        "service": "ml-acoustic-monitor",
        "model_available": inference_engine.is_available,
        "model_type": inference_engine.model_type or "None",
        "timestamp": time.time(),
    }


@app.get("/model-info")
def get_model_info():
    """Returns loaded model configuration and performance metrics."""
    return {
        "available": inference_engine.is_available,
        "model_type": inference_engine.model_type or "None",
        "config": inference_engine.config,
        "classes": inference_engine.class_names,
        "threshold": inference_engine.threshold,
    }


@app.post("/predict")
async def predict_anomaly(
    file: Optional[UploadFile] = File(None),
    audio_path: Optional[str] = Form(None),
    channel_index: Optional[int] = Form(None),
):
    """
    Evaluates ML anomaly score on the provided audio.
    Accepts either an uploaded WAV file or an absolute file path.
    """
    tmp_file = None
    try:
        if file is not None:
            suffix = Path(file.filename or "audio.wav").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                target_path = tmp.name
                tmp_file = target_path
        elif audio_path:
            target_path = audio_path
        else:
            raise HTTPException(status_code=400, detail="Must provide either audio file or audio_path")

        # Destination for rendered spectrogram image
        spec_dir = Path("backend/uploads/spectrograms")
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_image_name = f"spec_{int(time.time()*1000)}.png"
        spec_image_path = spec_dir / spec_image_name

        start_time = time.perf_counter()
        result = inference_engine.predict_file(
            target_path,
            channel_index=channel_index,
            save_spectrogram_to=str(spec_image_path),
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        result["processing_time_ms"] = round(latency_ms, 2)
        return result

    finally:
        if tmp_file and Path(tmp_file).exists():
            Path(tmp_file).unlink(missing_ok=True)


@app.post("/gate-and-predict")
async def gate_and_predict(
    file: Optional[UploadFile] = File(None),
    audio_path: Optional[str] = Form(None),
    gate_type: str = Form("amrag"),
    resource_level: float = Form(1.0),
    channel_index: Optional[int] = Form(None),
):
    """
    The Core Workflow:
    1. Edge Acoustic Gate processes raw audio
    2. If IGNORED -> ML execution is SKIPPED (saving compute/energy)
    3. If TRIGGERED -> audio is forwarded to downstream ML Anomaly Detection.
    """
    tmp_file = None
    try:
        if file is not None:
            suffix = Path(file.filename or "audio.wav").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                target_path = tmp.name
                tmp_file = target_path
        elif audio_path:
            target_path = audio_path
        else:
            raise HTTPException(status_code=400, detail="Must provide audio file or audio_path")

        # Select gate
        gtype = gate_type.lower().strip()
        if gtype in ("amrag", "adaptive", "proposed"):
            gate = AdaptiveMultiBandResourceGate(resource_level=resource_level)
        elif gtype in ("baseline_a", "amplitude"):
            gate = BaselineFixedAmplitudeGate(threshold=0.05, resource_level=resource_level)
        elif gtype in ("baseline_b", "band"):
            gate = BaselineFixedBandGate(target_band_name="mid", threshold=0.02, resource_level=resource_level)
        elif gtype in ("baseline_c", "snr"):
            gate = BaselineAdaptiveSNRGate(snr_threshold_db=6.0, resource_level=resource_level)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported gate_type: {gate_type}")

        gate_start = time.perf_counter()
        gate_report = gate.process_audio(target_path, channel_index=channel_index)
        gate_time_ms = (time.perf_counter() - gate_start) * 1000.0

        response = {
            "gate": {
                "gate_type": gate_report.gate_type,
                "gate_version": gate_report.gate_version,
                "resource_level": gate_report.resource_level,
                "duration_sec": round(gate_report.duration_sec, 3),
                "total_frames": gate_report.total_frames,
                "triggered_frames": gate_report.triggered_frames,
                "trigger_rate": round(gate_report.trigger_rate, 4),
                "overall_triggered": gate_report.overall_triggered,
                "max_trigger_score": round(gate_report.max_trigger_score, 4),
                "threshold": round(gate_report.threshold, 4),
                "events": gate_report.events,
                "metadata": gate_report.metadata,
                "gate_processing_time_ms": round(gate_time_ms, 2),
            },
            "ml_executed": False,
            "ml_prediction": None,
        }

        # Step 2: Triggered-only ML execution
        if gate_report.overall_triggered:
            spec_dir = Path("backend/uploads/spectrograms")
            spec_dir.mkdir(parents=True, exist_ok=True)
            spec_image_path = spec_dir / f"spec_{int(time.time()*1000)}.png"

            ml_start = time.perf_counter()
            ml_result = inference_engine.predict_file(
                target_path,
                channel_index=channel_index,
                save_spectrogram_to=str(spec_image_path),
            )
            ml_time_ms = (time.perf_counter() - ml_start) * 1000.0
            ml_result["processing_time_ms"] = round(ml_time_ms, 2)

            response["ml_executed"] = True
            response["ml_prediction"] = ml_result

        return response

    finally:
        if tmp_file and Path(tmp_file).exists():
            Path(tmp_file).unlink(missing_ok=True)
