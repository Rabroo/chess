#!/usr/bin/env python3
"""
Chess Position Evaluation Model Training

Trains models to predict Stockfish evaluation (centipawns) from FEN positions.
Includes 16 neural network architectures + 18 traditional ML models.

Models:
  Neural Networks: MLP, CNN, ResNet, Transformer, SE-ResNet, DenseNet, NNUE, etc.
  Traditional ML: Random Forest, XGBoost, LightGBM, Gradient Boosting, Ridge, etc.

Usage:
    python3 train_eval_model.py --data ./raw/chess --epochs 50 --model resnet
    python3 train_eval_model.py --data ./raw/chess --model random_forest
    python3 train_eval_model.py --list-models
    python3 train_eval_model.py --compare resnet,xgboost,lightgbm,transformer
"""

import argparse
import glob
import math
import os
import time
from pathlib import Path
from typing import Dict, Type

import numpy as np

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run: pip3 install torch")
    exit(1)

# Check for scikit-learn (for traditional ML models)
try:
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor,
        HistGradientBoostingRegressor
    )
    from sklearn.linear_model import (
        Ridge, Lasso, ElasticNet, SGDRegressor,
        BayesianRidge, HuberRegressor
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR, LinearSVR
    from sklearn.neural_network import MLPRegressor as SKLearnMLP
    from sklearn.kernel_ridge import KernelRidge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Traditional ML models unavailable.")
    print("Install with: pip3 install scikit-learn")

# Check for XGBoost (optional, often best performing)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Check for LightGBM (optional, very fast)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Check for CatBoost (optional, handles categorical well)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# Piece mappings for FEN parsing
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,    # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,  # Black pieces
}


def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Convert FEN string to 12x8x8 tensor.

    12 channels: P, N, B, R, Q, K, p, n, b, r, q, k
    Each channel is 8x8 binary (1 if piece present, 0 otherwise)
    """
    board = np.zeros((12, 8, 8), dtype=np.float32)

    # Parse only the piece placement part of FEN
    piece_placement = fen.split()[0]

    row = 0
    col = 0

    for char in piece_placement:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        elif char in PIECE_TO_INDEX:
            if row < 8 and col < 8:
                board[PIECE_TO_INDEX[char], row, col] = 1.0
            col += 1

    return board


def parse_score(score_str: str) -> float:
    """Parse score string to centipawns float."""
    score_str = score_str.strip()

    # Handle mate scores
    if score_str.startswith('M'):
        mate_in = int(score_str[1:])
        if mate_in > 0:
            return 10000 - mate_in * 10
        else:
            return -10000 - mate_in * 10

    # Regular centipawn score (stored as pawns, convert to centipawns)
    try:
        pawns = float(score_str)
        return pawns * 100
    except ValueError:
        return 0.0


class ChessDataset(Dataset):
    """Dataset for chess positions and evaluations."""

    def __init__(self, data_path: str, max_score: int = 15000):
        self.positions = []
        self.scores = []

        data_path = Path(data_path)
        loaded = 0
        skipped = 0

        # Check if it's a TSV file (fast) or directory (slow)
        if data_path.suffix == '.tsv' or data_path.suffix == '.csv':
            # Try to load from cache first
            cache_path = data_path.with_suffix(f'.cache_{max_score}.npz')
            if cache_path.exists():
                print(f"Loading from cache: {cache_path}")
                start = time.time()
                data = np.load(cache_path)
                self.positions = data['positions']
                self.scores = data['scores']
                self.score_scale = 500.0
                self.scores_normalized = self.scores / self.score_scale
                print(f"Loaded {len(self.positions):,} positions in {time.time() - start:.1f}s")
                return

            print(f"Loading from consolidated file: {data_path}")
            with open(data_path, 'r') as f:
                lines = f.readlines()

            total = len(lines)
            for i, line in enumerate(lines):
                if i % 100000 == 0:
                    print(f"\rProcessing: {i:,}/{total:,}", end="")

                try:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        skipped += 1
                        continue

                    fen = parts[0]
                    score_str = parts[1]

                    board_tensor = fen_to_tensor(fen)
                    score = parse_score(score_str)

                    if max_score > 0 and abs(score) > max_score:
                        skipped += 1
                        continue

                    self.positions.append(board_tensor)
                    self.scores.append(score)
                    loaded += 1

                except Exception:
                    skipped += 1
                    continue

            print()  # newline after progress
        else:
            # Legacy: load from individual files (slow)
            print(f"Loading from directory: {data_path} (slow - consider using TSV)")
            position_files = sorted(glob.glob(str(data_path / "position_*.txt")))
            position_files = [f for f in position_files if "_score" not in f and "_meta" not in f]

            for pos_file in position_files:
                score_file = pos_file.replace(".txt", "_score.txt")

                if not os.path.exists(score_file):
                    skipped += 1
                    continue

                try:
                    with open(pos_file, 'r') as f:
                        fen = f.read().strip()

                    with open(score_file, 'r') as f:
                        score_str = f.readline().strip()

                    board_tensor = fen_to_tensor(fen)
                    score = parse_score(score_str)

                    if max_score > 0 and abs(score) > max_score:
                        skipped += 1
                        continue

                    self.positions.append(board_tensor)
                    self.scores.append(score)
                    loaded += 1

                except Exception:
                    skipped += 1
                    continue

        print(f"Loaded {loaded:,} positions, skipped {skipped:,}")

        self.positions = np.array(self.positions)
        self.scores = np.array(self.scores, dtype=np.float32)

        self.score_scale = 500.0
        self.scores_normalized = self.scores / self.score_scale

        # Save cache for fast loading next time
        if data_path.suffix in ('.tsv', '.csv'):
            cache_path = data_path.with_suffix(f'.cache_{max_score}.npz')
            print(f"Saving cache to: {cache_path}")
            np.savez(cache_path, positions=self.positions, scores=self.scores)
            print("Cache saved - next run will load instantly")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.positions[idx]),
            torch.tensor(self.scores_normalized[idx], dtype=torch.float32)
        )


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class ChessModelBase(nn.Module):
    """Base class for all chess evaluation models."""

    description = "Base model"

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# MLP Models
# -----------------------------------------------------------------------------

class MLP_Small(ChessModelBase):
    """Small MLP - fast training, baseline."""

    description = "Small MLP (256 hidden) - 200K params, fast baseline"

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class MLP_Medium(ChessModelBase):
    """Medium MLP - balanced speed/accuracy."""

    description = "Medium MLP (512-256) - 400K params"

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class MLP_Large(ChessModelBase):
    """Large MLP - more capacity."""

    description = "Large MLP (1024-512-256) - 1M params"

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


# -----------------------------------------------------------------------------
# CNN Models
# -----------------------------------------------------------------------------

class CNN_Shallow(ChessModelBase):
    """Shallow CNN - 2 conv layers."""

    description = "Shallow CNN (2 layers) - 500K params, fast"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class CNN_Medium(ChessModelBase):
    """Medium CNN - 3 conv layers (original)."""

    description = "Medium CNN (3 layers) - 13M params"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)


class CNN_Deep(ChessModelBase):
    """Deep CNN - 5 conv layers."""

    description = "Deep CNN (5 layers) - 25M params"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# -----------------------------------------------------------------------------
# ResNet Models (AlphaZero/Lc0 style)
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.relu(x)
        return x


class ResNet_Small(ChessModelBase):
    """Small ResNet - 4 residual blocks."""

    description = "Small ResNet (4 blocks, 64ch) - 600K params, AlphaZero-style"

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)

        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class ResNet_Medium(ChessModelBase):
    """Medium ResNet - 8 residual blocks."""

    description = "Medium ResNet (8 blocks, 128ch) - 3M params"

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(128)

        self.blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(8)])

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class ResNet_Large(ChessModelBase):
    """Large ResNet - 16 residual blocks (Lc0-like)."""

    description = "Large ResNet (16 blocks, 256ch) - 20M params, Lc0-style"

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(256)

        self.blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(16)])

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# -----------------------------------------------------------------------------
# Transformer Models
# -----------------------------------------------------------------------------

class ChessTransformer_Small(ChessModelBase):
    """Small Transformer - treats squares as tokens."""

    description = "Small Transformer (4 layers, 128dim) - 1M params"

    def __init__(self, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.d_model = d_model

        # Project 12 piece channels to d_model
        self.input_proj = nn.Linear(12, d_model)

        # Positional encoding for 64 squares
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 12, 8, 8) -> (B, 64, 12)
        B = x.size(0)
        x = x.view(B, 12, 64).permute(0, 2, 1)

        # Project to d_model
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


class ChessTransformer_Medium(ChessModelBase):
    """Medium Transformer - more capacity."""

    description = "Medium Transformer (6 layers, 256dim) - 5M params"

    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(12, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 12, 64).permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


class ChessTransformer_Large(ChessModelBase):
    """Large Transformer - maximum capacity."""

    description = "Large Transformer (8 layers, 512dim) - 20M params"

    def __init__(self, d_model=512, nhead=8, num_layers=8):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(12, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 12, 64).permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# -----------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) Models
# -----------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        # Squeeze
        y = x.view(B, C, -1).mean(dim=2)
        # Excitation
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        # Scale
        y = y.view(B, C, 1, 1)
        return x * y


class SE_ResNet(ChessModelBase):
    """ResNet with Squeeze-and-Excitation blocks."""

    description = "SE-ResNet (8 blocks with channel attention) - 4M params"

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(128)

        self.blocks = nn.ModuleList()
        for _ in range(8):
            self.blocks.append(nn.Sequential(
                ResidualBlock(128),
                SEBlock(128)
            ))

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# -----------------------------------------------------------------------------
# DenseNet-style Model
# -----------------------------------------------------------------------------

class DenseBlock(nn.Module):
    """Dense block with concatenated features."""

    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return torch.cat([x, out], dim=1)


class DenseNet(ChessModelBase):
    """DenseNet-style with concatenated features."""

    description = "DenseNet (6 dense blocks) - 2M params, efficient"

    def __init__(self):
        super().__init__()
        growth_rate = 32

        self.conv_in = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)

        # Dense blocks
        channels = 64
        self.blocks = nn.ModuleList()
        for _ in range(6):
            self.blocks.append(DenseBlock(channels, growth_rate))
            channels += growth_rate

        self.fc1 = nn.Linear(channels * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# -----------------------------------------------------------------------------
# Hybrid CNN-Attention Model
# -----------------------------------------------------------------------------

class CNN_Attention(ChessModelBase):
    """CNN with self-attention layer."""

    description = "CNN + Attention (conv then attention) - 8M params"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # Self-attention over spatial locations
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Reshape for attention: (B, 256, 8, 8) -> (B, 64, 256)
        B = x.size(0)
        x = x.view(B, 256, 64).permute(0, 2, 1)

        # Self-attention
        x, _ = self.attention(x, x, x)

        # Global average pool
        x = x.mean(dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# -----------------------------------------------------------------------------
# NNUE-style Model (Stockfish architecture)
# -----------------------------------------------------------------------------

class NNUE_Style(ChessModelBase):
    """NNUE-style network (like Stockfish)."""

    description = "NNUE-style (sparse input, efficient) - 500K params, fast inference"

    def __init__(self):
        super().__init__()
        # NNUE uses a different input encoding, but we'll approximate
        # with the 12x64 = 768 input features

        self.fc1 = nn.Linear(12 * 64, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

        # ClippedReLU is used in NNUE
        self.clipped_relu = lambda x: torch.clamp(x, 0, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.clipped_relu(self.fc1(x))
        x = self.clipped_relu(self.fc2(x))
        x = self.clipped_relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, Type[ChessModelBase]] = {
    # MLP
    "mlp_small": MLP_Small,
    "mlp_medium": MLP_Medium,
    "mlp_large": MLP_Large,

    # CNN
    "cnn_shallow": CNN_Shallow,
    "cnn_medium": CNN_Medium,
    "cnn_deep": CNN_Deep,

    # ResNet
    "resnet_small": ResNet_Small,
    "resnet_medium": ResNet_Medium,
    "resnet_large": ResNet_Large,

    # Transformer
    "transformer_small": ChessTransformer_Small,
    "transformer_medium": ChessTransformer_Medium,
    "transformer_large": ChessTransformer_Large,

    # Special architectures
    "se_resnet": SE_ResNet,
    "densenet": DenseNet,
    "cnn_attention": CNN_Attention,
    "nnue": NNUE_Style,
}

# Aliases for convenience
MODEL_REGISTRY["mlp"] = MLP_Medium
MODEL_REGISTRY["cnn"] = CNN_Medium
MODEL_REGISTRY["resnet"] = ResNet_Medium
MODEL_REGISTRY["transformer"] = ChessTransformer_Medium

# Optimal epochs per model (tuned for 1M positions, depth 60)
# Smaller/simpler models need more epochs, larger models converge faster
OPTIMAL_EPOCHS: Dict[str, int] = {
    # MLP - simple architecture, needs more epochs
    "mlp_small": 60,
    "mlp_medium": 50,
    "mlp_large": 40,

    # CNN - moderate convergence
    "cnn_shallow": 45,
    "cnn_medium": 35,
    "cnn_deep": 30,

    # ResNet - efficient learning with residuals
    "resnet_small": 40,
    "resnet_medium": 35,
    "resnet_large": 30,

    # Transformer - needs more epochs for attention to learn
    "transformer_small": 50,
    "transformer_medium": 45,
    "transformer_large": 40,

    # Special architectures
    "se_resnet": 35,      # SE blocks help convergence
    "densenet": 40,       # Dense connections need time
    "cnn_attention": 35,  # Hybrid converges well
    "nnue": 70,           # Simple arch, needs more epochs
}

# Aliases
OPTIMAL_EPOCHS["mlp"] = OPTIMAL_EPOCHS["mlp_medium"]
OPTIMAL_EPOCHS["cnn"] = OPTIMAL_EPOCHS["cnn_medium"]
OPTIMAL_EPOCHS["resnet"] = OPTIMAL_EPOCHS["resnet_medium"]
OPTIMAL_EPOCHS["transformer"] = OPTIMAL_EPOCHS["transformer_medium"]


# =============================================================================
# TRADITIONAL ML MODELS (scikit-learn)
# =============================================================================

TRADITIONAL_MODELS: Dict[str, dict] = {}

if SKLEARN_AVAILABLE:
    # === Tree-based Ensembles ===
    TRADITIONAL_MODELS["random_forest"] = {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 100, "max_depth": 12, "n_jobs": -1, "random_state": 42, "verbose": 1},
        "description": "Random Forest (100 trees, depth 12) - ensemble baseline",
        "eta_minutes": 10,
    }
    TRADITIONAL_MODELS["extra_trees"] = {
        "class": ExtraTreesRegressor,
        "params": {"n_estimators": 100, "max_depth": 12, "n_jobs": -1, "random_state": 42, "verbose": 1},
        "description": "Extra Trees (100 trees) - random splits, faster",
        "eta_minutes": 5,
    }
    TRADITIONAL_MODELS["gradient_boosting"] = {
        "class": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42, "verbose": 1},
        "description": "Gradient Boosting (100 trees) - sequential ensemble",
        "eta_minutes": 45,
    }
    TRADITIONAL_MODELS["hist_gradient_boosting"] = {
        "class": HistGradientBoostingRegressor,
        "params": {"max_iter": 100, "max_depth": 10, "learning_rate": 0.1, "random_state": 42, "verbose": 1},
        "description": "Histogram Gradient Boosting - fast, LightGBM-like",
        "eta_minutes": 2,
    }
    TRADITIONAL_MODELS["adaboost"] = {
        "class": AdaBoostRegressor,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
        "description": "AdaBoost (100 estimators) - adaptive boosting",
        "eta_minutes": 30,
    }
    TRADITIONAL_MODELS["bagging"] = {
        "class": BaggingRegressor,
        "params": {"n_estimators": 50, "n_jobs": -1, "random_state": 42, "verbose": 1},
        "description": "Bagging (50 estimators) - bootstrap aggregating",
        "eta_minutes": 15,
    }
    TRADITIONAL_MODELS["decision_tree"] = {
        "class": DecisionTreeRegressor,
        "params": {"max_depth": 20, "random_state": 42},
        "description": "Decision Tree (depth 20) - single tree baseline",
        "eta_minutes": 1,
    }

    # === Linear Models ===
    TRADITIONAL_MODELS["ridge"] = {
        "class": Ridge,
        "params": {"alpha": 1.0},
        "description": "Ridge Regression (L2) - linear baseline",
        "eta_minutes": 0.1,
    }
    TRADITIONAL_MODELS["lasso"] = {
        "class": Lasso,
        "params": {"alpha": 0.1, "max_iter": 1000},
        "description": "Lasso Regression (L1) - sparse linear",
        "eta_minutes": 0.2,
    }
    TRADITIONAL_MODELS["elastic_net"] = {
        "class": ElasticNet,
        "params": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 1000},
        "description": "ElasticNet (L1+L2) - combined regularization",
        "eta_minutes": 0.2,
    }
    TRADITIONAL_MODELS["bayesian_ridge"] = {
        "class": BayesianRidge,
        "params": {},
        "description": "Bayesian Ridge - probabilistic linear",
        "eta_minutes": 0.5,
    }
    TRADITIONAL_MODELS["huber"] = {
        "class": HuberRegressor,
        "params": {"max_iter": 200},
        "description": "Huber Regressor - robust to outliers",
        "eta_minutes": 1,
    }
    TRADITIONAL_MODELS["sgd"] = {
        "class": SGDRegressor,
        "params": {"max_iter": 1000, "tol": 1e-3, "random_state": 42},
        "description": "SGD Regressor - stochastic gradient descent",
        "eta_minutes": 0.2,
    }

    # === Instance-based ===
    TRADITIONAL_MODELS["knn"] = {
        "class": KNeighborsRegressor,
        "params": {"n_neighbors": 5, "weights": "distance", "n_jobs": -1},
        "description": "K-Nearest Neighbors (k=5) - instance-based",
        "eta_minutes": 5,
    }

    # === Support Vector Machines ===
    TRADITIONAL_MODELS["linear_svr"] = {
        "class": LinearSVR,
        "params": {"max_iter": 1000, "random_state": 42},
        "description": "Linear SVR - fast linear SVM",
        "eta_minutes": 2,
    }
    # Note: SVR with RBF kernel is too slow for 1M samples

    # === sklearn Neural Network ===
    TRADITIONAL_MODELS["sklearn_mlp"] = {
        "class": SKLearnMLP,
        "params": {"hidden_layer_sizes": (256, 128), "max_iter": 200,
                   "early_stopping": True, "random_state": 42, "verbose": True},
        "description": "sklearn MLP (256-128) - sklearn neural network",
        "eta_minutes": 10,
    }

# === External Libraries ===
if XGBOOST_AVAILABLE:
    TRADITIONAL_MODELS["xgboost"] = {
        "class": xgb.XGBRegressor,
        "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                   "n_jobs": -1, "random_state": 42, "tree_method": "hist", "verbosity": 1},
        "description": "XGBoost (100 trees) - often best traditional ML",
        "eta_minutes": 2,
    }

if LIGHTGBM_AVAILABLE:
    TRADITIONAL_MODELS["lightgbm"] = {
        "class": lgb.LGBMRegressor,
        "params": {"n_estimators": 100, "max_depth": 10, "learning_rate": 0.1,
                   "n_jobs": -1, "random_state": 42, "verbose": 1},
        "description": "LightGBM (100 trees) - very fast gradient boosting",
        "eta_minutes": 1,
    }

if CATBOOST_AVAILABLE:
    TRADITIONAL_MODELS["catboost"] = {
        "class": CatBoostRegressor,
        "params": {"iterations": 100, "depth": 6, "learning_rate": 0.1,
                   "random_state": 42, "verbose": 10},
        "description": "CatBoost (100 trees) - handles categoricals well",
        "eta_minutes": 2,
    }


def train_traditional_model(model_name, X_train, y_train, X_val, y_val, score_scale):
    """Train a traditional ML model and return metrics."""
    if model_name not in TRADITIONAL_MODELS:
        raise ValueError(f"Unknown traditional model: {model_name}")

    model_info = TRADITIONAL_MODELS[model_name]
    model_class = model_info["class"]
    params = model_info["params"]
    eta_minutes = model_info.get("eta_minutes", 5)

    if eta_minutes >= 1:
        print(f"Training {model_name}... (estimated: ~{eta_minutes:.0f} min)", flush=True)
    else:
        print(f"Training {model_name}... (estimated: <1 min)", flush=True)

    start_time = time.time()
    model = model_class(**params)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.1f}s", flush=True)

    # Validation metrics
    val_pred = model.predict(X_val)
    val_mae = np.mean(np.abs(val_pred - y_val)) * score_scale
    print(f"Validation MAE: {val_mae:.1f} cp", flush=True)

    return model, val_mae


def evaluate_traditional_model(model, X_test, y_test, score_scale):
    """Evaluate a traditional ML model."""
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

    predictions_cp = predictions * score_scale
    actuals_cp = y_test * score_scale

    mae = np.mean(np.abs(predictions_cp - actuals_cp))
    rmse = np.sqrt(np.mean((predictions_cp - actuals_cp) ** 2))
    correlation = np.corrcoef(predictions_cp, actuals_cp)[0, 1]

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "inference_time_ms": inference_time,
        "throughput": 1000 / inference_time if inference_time > 0 else 0,
    }


def list_models():
    """Print all available models."""
    print("\nAvailable Models:")
    print("=" * 70)

    categories = {
        "MLP (Multi-Layer Perceptron)": ["mlp_small", "mlp_medium", "mlp_large"],
        "CNN (Convolutional Neural Network)": ["cnn_shallow", "cnn_medium", "cnn_deep"],
        "ResNet (Residual Network - AlphaZero/Lc0 style)": ["resnet_small", "resnet_medium", "resnet_large"],
        "Transformer (Attention-based)": ["transformer_small", "transformer_medium", "transformer_large"],
        "Special Architectures": ["se_resnet", "densenet", "cnn_attention", "nnue"],
    }

    for category, models in categories.items():
        print(f"\n{category}:")
        print("-" * 70)
        for name in models:
            model_class = MODEL_REGISTRY[name]
            print(f"  {name:20} - {model_class.description}")

    # Traditional ML models
    if TRADITIONAL_MODELS:
        print(f"\nTraditional ML (scikit-learn):")
        print("-" * 70)
        for name, info in TRADITIONAL_MODELS.items():
            print(f"  {name:20} - {info['description']}")

    print("\n" + "=" * 70)
    print("Aliases: mlp, cnn, resnet, transformer (point to medium variants)")
    print("\nUsage:")
    print("  python3 train_eval_model.py --model resnet_small           # Uses optimal epochs (40)")
    print("  python3 train_eval_model.py --compare mlp,cnn,resnet,transformer  # Auto epochs per model")
    print("  python3 train_eval_model.py --model cnn_deep --epochs 50   # Override with custom epochs")
    print("  python3 train_eval_model.py --compare random_forest,xgboost,ridge  # Traditional ML")
    print("  python3 train_eval_model.py --compare resnet,xgboost,transformer   # Mix NN and traditional")


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, epochs, device, score_scale, model_name="model", resume=False, use_amp=False):
    """Train the model with checkpoint support and optional mixed precision."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Mixed precision setup
    use_amp = use_amp and device.type in ('cuda', 'mps')
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16

    if use_amp:
        print(f"Using automatic mixed precision ({amp_dtype})")

    best_val_loss = float('inf')
    best_model_path = f'best_{model_name}.pth'
    checkpoint_path = f'checkpoint_{model_name}.pth'
    start_epoch = 0

    # Resume from checkpoint if exists
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx}/{num_batches}", end="", flush=True)

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                val_mae += torch.abs(outputs - batch_y).mean().item() * score_scale

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        # Save checkpoint every 5 epochs and on last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.1f} cp", flush=True)

    return best_val_loss, best_model_path


def evaluate_model(model, test_loader, device, score_scale):
    """Evaluate model and return metrics."""
    model.eval()

    predictions = []
    actuals = []
    inference_times = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            start = time.perf_counter()
            outputs = model(batch_x)
            inference_times.append((time.perf_counter() - start) / len(batch_x))

            predictions.extend((outputs.cpu().numpy() * score_scale).tolist())
            actuals.extend((batch_y.numpy() * score_scale).tolist())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    avg_inference_time = np.mean(inference_times) * 1000

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "inference_time_ms": avg_inference_time,
        "throughput": 1000 / avg_inference_time,
    }


def print_evaluation(metrics, model_name="Model"):
    """Print evaluation results."""
    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 50)
    print(f"Mean Absolute Error: {metrics['mae']:.1f} centipawns ({metrics['mae']/100:.2f} pawns)")
    print(f"RMSE: {metrics['rmse']:.1f} centipawns")
    print(f"Correlation with Stockfish: {metrics['correlation']:.4f}")
    print(f"Avg inference time: {metrics['inference_time_ms']:.3f} ms per position")
    print(f"Throughput: {metrics['throughput']:.0f} positions/second")


def main():
    parser = argparse.ArgumentParser(description="Train chess evaluation model")
    parser.add_argument("--data", "-d", type=str, default="./raw/chess",
                        help="Directory containing position files")
    parser.add_argument("--epochs", "-e", type=int, default=None,
                        help="Number of training epochs (default: auto per model)")
    parser.add_argument("--batch-size", "-b", type=int, default=128,
                        help="Batch size for training (default: 128, use 256+ for faster training)")
    parser.add_argument("--model", "-m", type=str, default="resnet_medium",
                        help="Model architecture (see --list-models)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare multiple models (comma-separated)")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run evaluation on existing model")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--max-score", type=int, default=15000,
                        help="Max score in centipawns to include (0=no limit, default: 15000)")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision (faster on CUDA/MPS)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size for quick testing (e.g., --max-samples 10000)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: use 10k samples to verify everything works")

    args = parser.parse_args()

    # Test mode uses small sample size
    if args.test:
        args.max_samples = 10000
        print("TEST MODE: Using 10,000 samples for quick verification\n")

    if args.list_models:
        list_models()
        return 0

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    print(f"\nLoading data from {args.data}...")
    print(f"Max score filter: {args.max_score} cp" if args.max_score > 0 else "No score filter")
    dataset = ChessDataset(args.data, max_score=args.max_score)

    if len(dataset) == 0:
        print("No data found! Make sure positions are scored.")
        return 1

    # Limit samples if requested
    if args.max_samples and len(dataset) > args.max_samples:
        print(f"Limiting dataset from {len(dataset):,} to {args.max_samples:,} samples")
        dataset.positions = dataset.positions[:args.max_samples]
        dataset.scores = dataset.scores[:args.max_samples]
        dataset.scores_normalized = dataset.scores_normalized[:args.max_samples]

    print(f"Total positions: {len(dataset)}")

    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Data loaders with optimizations
    # - num_workers: parallel data loading (0 for MPS compatibility)
    # - pin_memory: faster CPU→GPU transfer (only for CUDA)
    # - persistent_workers: keep workers alive between epochs
    num_workers = 0 if device.type == 'mps' else 4
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    if num_workers > 0:
        print(f"DataLoader: {num_workers} workers, pin_memory={pin_memory}")

    # Compare multiple models
    if args.compare:
        model_names = [m.strip() for m in args.compare.split(",")]
        results = []

        # Check if any traditional models are requested
        has_traditional = any(m in TRADITIONAL_MODELS for m in model_names)

        # Prepare numpy arrays for traditional ML models (only if needed)
        if has_traditional:
            print("\nPreparing data for traditional ML models...")
            # Get indices from the splits
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices
            test_indices = test_dataset.indices

            # Flatten positions to (N, 768) for traditional ML
            X_train = dataset.positions[train_indices].reshape(len(train_indices), -1)
            y_train = dataset.scores_normalized[train_indices]
            X_val = dataset.positions[val_indices].reshape(len(val_indices), -1)
            y_val = dataset.scores_normalized[val_indices]
            X_test = dataset.positions[test_indices].reshape(len(test_indices), -1)
            y_test = dataset.scores_normalized[test_indices]
            print(f"Traditional ML data prepared: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, {X_test.shape[0]:,} test")

        for model_name in model_names:
            # Check if it's a traditional ML model
            if model_name in TRADITIONAL_MODELS:
                print(f"\n{'#' * 60}")
                print(f"Training: {model_name} (Traditional ML)")
                print("#" * 60)

                model_info = TRADITIONAL_MODELS[model_name]
                print(f"Description: {model_info['description']}")

                # Train traditional model
                model, val_mae = train_traditional_model(
                    model_name, X_train, y_train, X_val, y_val, dataset.score_scale
                )

                # Evaluate
                metrics = evaluate_traditional_model(model, X_test, y_test, dataset.score_scale)
                metrics["model"] = model_name
                metrics["params"] = "N/A"
                results.append(metrics)
                continue

            # Neural network model
            if model_name not in MODEL_REGISTRY:
                print(f"Unknown model: {model_name}")
                continue

            print(f"\n{'#' * 60}")
            print(f"Training: {model_name}")
            print("#" * 60)

            model_class = MODEL_REGISTRY[model_name]
            model = model_class().to(device)

            print(f"Description: {model_class.description}")
            print(f"Parameters: {model.count_parameters():,}")

            # Get optimal epochs for this model
            epochs = args.epochs if args.epochs else OPTIMAL_EPOCHS.get(model_name, 50)
            print(f"Epochs: {epochs} {'(auto)' if not args.epochs else ''}")

            # Train
            val_loss, model_path = train_model(
                model, train_loader, val_loader, epochs, device,
                dataset.score_scale, model_name, resume=args.resume, use_amp=args.amp
            )

            # Load best and evaluate
            model.load_state_dict(torch.load(model_path, map_location=device))
            metrics = evaluate_model(model, test_loader, device, dataset.score_scale)
            metrics["model"] = model_name
            metrics["params"] = model.count_parameters()
            results.append(metrics)

        # Print comparison table
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"{'Model':<20} {'Params':>12} {'MAE (cp)':>10} {'Corr':>8} {'Speed':>12}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: x['mae']):
            params_str = f"{r['params']:,}" if isinstance(r['params'], int) else r['params']
            print(f"{r['model']:<20} {params_str:>12} {r['mae']:>10.1f} {r['correlation']:>8.4f} {r['throughput']:>10.0f}/s")

        return 0

    # Single model training - check if traditional ML
    if args.model in TRADITIONAL_MODELS:
        print(f"\nModel: {args.model} (Traditional ML)")
        print(f"Description: {TRADITIONAL_MODELS[args.model]['description']}")

        # Prepare numpy arrays
        print("\nPreparing data for traditional ML...")
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices

        X_train = dataset.positions[train_indices].reshape(len(train_indices), -1)
        y_train = dataset.scores_normalized[train_indices]
        X_val = dataset.positions[val_indices].reshape(len(val_indices), -1)
        y_val = dataset.scores_normalized[val_indices]
        X_test = dataset.positions[test_indices].reshape(len(test_indices), -1)
        y_test = dataset.scores_normalized[test_indices]

        # Train
        model, val_mae = train_traditional_model(
            args.model, X_train, y_train, X_val, y_val, dataset.score_scale
        )

        # Evaluate
        metrics = evaluate_traditional_model(model, X_test, y_test, dataset.score_scale)
        print_evaluation(metrics, args.model)

        return 0

    # Single neural network model training
    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model: {args.model}")
        print("Use --list-models to see available models")
        return 1

    model_class = MODEL_REGISTRY[args.model]
    model = model_class().to(device)

    print(f"\nModel: {args.model}")
    print(f"Description: {model_class.description}")
    print(f"Parameters: {model.count_parameters():,}")

    if args.test_only:
        model_path = f'best_{args.model}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded existing model: {model_path}")
        else:
            print(f"No model found: {model_path}")
            return 1
    else:
        # Get optimal epochs for this model
        epochs = args.epochs if args.epochs else OPTIMAL_EPOCHS.get(args.model, 50)
        print(f"\nTraining for {epochs} epochs {'(auto-selected)' if not args.epochs else ''}...")
        print("-" * 50)
        _, model_path = train_model(
            model, train_loader, val_loader, epochs, device,
            dataset.score_scale, args.model, resume=args.resume, use_amp=args.amp
        )
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, dataset.score_scale)
    print_evaluation(metrics, args.model)

    print(f"\nModel saved to: best_{args.model}.pth")

    return 0


if __name__ == "__main__":
    exit(main())
