"""
Machine Anomaly 2D CNN Architecture.

Implements the standard convolutional network specified in README Section 14:
Conv2D -> BatchNorm -> ReLU -> MaxPool
Conv2D -> BatchNorm -> ReLU -> MaxPool
Conv2D -> GlobalAveragePooling -> Dense -> Dropout -> Output
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MachineAcousticCNN(nn.Module):
    """
    Lightweight 2D CNN for spectrogram machine acoustic anomaly classification and feature embedding.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        # Input shape: (Batch, 1, n_mels=128, n_frames)
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(64, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 128-dimensional embedding vector after Global Average Pooling."""
        # x: (B, 1, M, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class logits."""
        features = self.extract_features(x)
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits


class SpectrogramAutoencoder(nn.Module):
    """
    Unsupervised Convolutional Autoencoder for Normal-Reference Anomaly Detection.
    Trained ONLY on normal machine operating audio.
    Anomaly score = Reconstruction MSE between original spectrogram and reconstructed output.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128)

        # Decoder
        self.dec_tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_tconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dec_bn1(self.dec_tconv1(z)))
        x = F.relu(self.dec_bn2(self.dec_tconv2(x)))
        x = torch.sigmoid(self.dec_tconv3(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        z = self.encode(x)
        recon = self.decode(z)
        # Crop or pad to match exact original shape if padding differences arise
        if recon.shape != orig_shape:
            recon = F.interpolate(recon, size=(orig_shape[2], orig_shape[3]), mode="bilinear", align_corners=False)
        return recon

    def compute_anomaly_score(self, x: torch.Tensor) -> float:
        """Computes reconstruction Mean Squared Error (MSE)."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            mse = F.mse_loss(recon, x, reduction="mean")
            return float(mse.item())
