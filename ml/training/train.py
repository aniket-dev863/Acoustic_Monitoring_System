"""
Model Training Script for Machine Acoustic Condition Monitoring.

Supports:
1. Unsupervised Autoencoder training on normal operational recordings (DCASE 2026 style)
2. Supervised 2D CNN training
3. Supervised EfficientNet fine-tuning

Saves all expected artifacts into ml/weights/:
- model.pt
- class_names.json
- config.json
- metrics.json
- training_history.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.cnn import MachineAcousticCNN, SpectrogramAutoencoder
from ml.models.efficientnet import AcousticEfficientNet
from ml.training.dataset import MachineAudioDataset, build_manifest_from_directory


def train_model(
    data_dir: str = "data/sample",
    output_dir: str = "ml/weights",
    model_type: str = "autoencoder",  # "autoencoder", "cnn", or "efficientnet"
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 0.001,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Training] Discovering audio samples in {data_dir}...")
    manifest = build_manifest_from_directory(data_dir)
    if len(manifest) == 0:
        raise RuntimeError(f"No audio files found in {data_dir}. Cannot train.")

    is_unsupervised = model_type.lower() == "autoencoder"
    dataset = MachineAudioDataset(manifest, is_unsupervised=is_unsupervised)
    if len(dataset) == 0:
        raise RuntimeError("No eligible samples found for dataset.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Training] Model: {model_type.upper()} | Samples: {len(dataset)} | Device: {device} | Epochs: {epochs}")

    # Instantiate model
    if is_unsupervised:
        model = SpectrogramAutoencoder().to(device)
        criterion = nn.MSELoss()
    elif model_type.lower() == "efficientnet":
        model = AcousticEfficientNet(num_classes=2, pretrained=False).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = MachineAcousticCNN(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {"train_loss": []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for tensors, labels, _ in dataloader:
            tensors = tensors.to(device)
            optimizer.zero_grad()

            if is_unsupervised:
                reconstructed = model(tensors)
                loss = criterion(reconstructed, tensors)
            else:
                labels = labels.to(device)
                logits = model(tensors)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(tensors)

        avg_loss = epoch_loss / len(dataset)
        history["train_loss"].append(round(avg_loss, 5))
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {avg_loss:.5f}")

    # Compute calibrated threshold for Autoencoder or baseline threshold
    calibrated_threshold = 0.035
    if is_unsupervised:
        model.eval()
        recon_errors = []
        with torch.no_grad():
            for tensors, _, _ in dataloader:
                tensors = tensors.to(device)
                for i in range(len(tensors)):
                    single = tensors[i : i + 1]
                    err = model.compute_anomaly_score(single)
                    recon_errors.append(err)
        # Set threshold at 95th percentile of normal reconstruction error
        if recon_errors:
            calibrated_threshold = float(np.percentile(recon_errors, 95)) * 1.5
            print(f"[Training] Calibrated Anomaly Threshold: {calibrated_threshold:.5f}")

    # Save artifacts
    weights_path = out_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"[Training] Saved model weights to {weights_path}")

    config = {
        "model_type": model_type,
        "version": "v1.0",
        "is_unsupervised": is_unsupervised,
        "threshold": calibrated_threshold,
        "sample_rate": 16000,
        "n_mels": 128,
        "epochs": epochs,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    class_names = ["normal", "anomaly"]
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    metrics = {
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "calibrated_threshold": calibrated_threshold,
        "total_training_samples": len(dataset),
        "status": "TRAINED",
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("[Training] All model artifacts saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Acoustic Anomaly Detection Model")
    parser.add_argument("--data_dir", type=str, default="data/sample")
    parser.add_argument("--model_type", type=str, default="autoencoder", choices=["autoencoder", "cnn", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, model_type=args.model_type, epochs=args.epochs)
