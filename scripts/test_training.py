#!/usr/bin/env python3
"""
Test suite for chess evaluation training system.
Run: python3 scripts/test_training.py
"""

import sys
import numpy as np

# Track test results
passed = 0
failed = 0
errors = []

def test(name, condition, error_msg=""):
    global passed, failed, errors
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1
        if error_msg:
            errors.append(f"{name}: {error_msg}")

def test_section(name):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)


# =============================================================================
# 1. IMPORTS
# =============================================================================
test_section("Imports")

try:
    import torch
    test("PyTorch import", True)
    test(f"PyTorch version ({torch.__version__})", True)
except ImportError as e:
    test("PyTorch import", False, str(e))
    print("\nCRITICAL: PyTorch required. Run: pip3 install torch")
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    SKLEARN_AVAILABLE = True
    test("scikit-learn import", True)
except ImportError:
    SKLEARN_AVAILABLE = False
    test("scikit-learn import", False, "pip3 install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    test("XGBoost import", True)
except ImportError:
    XGBOOST_AVAILABLE = False
    test("XGBoost import (optional)", True)  # Optional, so pass anyway
    print("       Note: XGBoost not installed (optional)")


# =============================================================================
# 2. DEVICE DETECTION
# =============================================================================
test_section("Device Detection")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    test("Apple Silicon GPU (MPS) available", True)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    test(f"NVIDIA GPU (CUDA) available: {torch.cuda.get_device_name(0)}", True)
else:
    device = torch.device("cpu")
    test("CPU fallback", True)
    print("       Note: No GPU detected, training will be slower")

print(f"       Using device: {device}")


# =============================================================================
# 3. FEN PARSING
# =============================================================================
test_section("FEN Parsing")

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

def fen_to_tensor(fen: str) -> np.ndarray:
    board = np.zeros((12, 8, 8), dtype=np.float32)
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

# Test starting position
start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
tensor = fen_to_tensor(start_fen)
test("FEN to tensor shape", tensor.shape == (12, 8, 8))
test("FEN tensor dtype", tensor.dtype == np.float32)
test("White pawns on rank 2", tensor[0, 6, :].sum() == 8)  # 8 white pawns
test("Black pawns on rank 7", tensor[6, 1, :].sum() == 8)  # 8 black pawns
test("White king on e1", tensor[5, 7, 4] == 1.0)
test("Black king on e8", tensor[11, 0, 4] == 1.0)

# Test empty board
empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
empty_tensor = fen_to_tensor(empty_fen)
test("Empty board sums to 0", empty_tensor.sum() == 0)


# =============================================================================
# 4. SCORE PARSING
# =============================================================================
test_section("Score Parsing")

def parse_score(score_str: str) -> float:
    score_str = score_str.strip()
    if score_str.startswith('M'):
        mate_in = int(score_str[1:])
        if mate_in > 0:
            return 10000 - mate_in * 10
        else:
            return -10000 - mate_in * 10
    try:
        pawns = float(score_str)
        return pawns * 100
    except ValueError:
        return 0.0

test("Parse positive score (1.5)", abs(parse_score("1.5") - 150.0) < 0.01)
test("Parse negative score (-2.3)", abs(parse_score("-2.3") - (-230.0)) < 0.01)
test("Parse zero score", parse_score("0") == 0.0)
test("Parse mate in 3", abs(parse_score("M3") - 9970.0) < 0.01)
test("Parse mate in -5", abs(parse_score("M-5") - (-9950.0)) < 0.01)


# =============================================================================
# 5. NEURAL NETWORK MODELS
# =============================================================================
test_section("Neural Network Models")

# Import all model classes
from train_eval_model import MODEL_REGISTRY, ChessModelBase

test(f"Model registry loaded ({len(MODEL_REGISTRY)} models)", len(MODEL_REGISTRY) >= 16)

# Test each model can be instantiated and do forward pass
sample_input = torch.randn(2, 12, 8, 8).to(device)  # Batch of 2

for model_name, model_class in MODEL_REGISTRY.items():
    if model_name in ['mlp', 'cnn', 'resnet', 'transformer']:  # Skip aliases
        continue
    try:
        model = model_class().to(device)
        with torch.no_grad():
            output = model(sample_input)

        correct_shape = output.shape == (2,)
        params = model.count_parameters()
        test(f"{model_name} forward pass ({params:,} params)", correct_shape)
    except Exception as e:
        test(f"{model_name} forward pass", False, str(e))


# =============================================================================
# 6. TRADITIONAL ML MODELS
# =============================================================================
test_section("Traditional ML Models")

if SKLEARN_AVAILABLE:
    from train_eval_model import TRADITIONAL_MODELS

    test(f"Traditional models loaded ({len(TRADITIONAL_MODELS)} models)", len(TRADITIONAL_MODELS) >= 4)

    # Test each traditional model can be instantiated and fit small data
    X_sample = np.random.randn(100, 768).astype(np.float32)
    y_sample = np.random.randn(100).astype(np.float32)

    for model_name, model_info in TRADITIONAL_MODELS.items():
        try:
            model = model_info['class'](**model_info['params'])
            model.fit(X_sample[:80], y_sample[:80])
            pred = model.predict(X_sample[80:])
            test(f"{model_name} fit and predict", len(pred) == 20)
        except Exception as e:
            test(f"{model_name} fit and predict", False, str(e))
else:
    print("  [SKIP] Traditional ML tests (scikit-learn not installed)")


# =============================================================================
# 7. DATA LOADING
# =============================================================================
test_section("Data Loading")

import os
tsv_path = "./raw/chess_quality.tsv"

if os.path.exists(tsv_path):
    test("TSV file exists", True)

    # Check file is readable and has correct format
    with open(tsv_path, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split('\t')

    test("TSV has tab-separated values", len(parts) >= 2)
    test("First column looks like FEN", '/' in parts[0])

    # Try to parse first line
    try:
        tensor = fen_to_tensor(parts[0])
        score = parse_score(parts[1])
        test("First position parseable", True)
    except Exception as e:
        test("First position parseable", False, str(e))

    # Count lines (estimate)
    with open(tsv_path, 'r') as f:
        line_count = sum(1 for _ in f)
    test(f"Dataset size ({line_count:,} positions)", line_count > 0)

    if line_count < 10000:
        print("       Warning: Small dataset may affect training quality")
else:
    test("TSV file exists", False, f"Not found: {tsv_path}")
    print("       Run: python3 consolidate.py")


# =============================================================================
# 8. TRAINING FUNCTIONS
# =============================================================================
test_section("Training Functions")

from train_eval_model import train_model, evaluate_model, train_traditional_model, evaluate_traditional_model

test("train_model function exists", callable(train_model))
test("evaluate_model function exists", callable(evaluate_model))
test("train_traditional_model function exists", callable(train_traditional_model))
test("evaluate_traditional_model function exists", callable(evaluate_traditional_model))


# =============================================================================
# 9. MEMORY CHECK
# =============================================================================
test_section("Memory Check")

# Test that we can allocate tensors for a batch
try:
    batch_size = 32
    test_batch = torch.randn(batch_size, 12, 8, 8).to(device)
    test("Can allocate batch tensor on device", True)
    del test_batch
except Exception as e:
    test("Can allocate batch tensor on device", False, str(e))

# Check if large models might have memory issues (MPS specific)
if device.type == 'mps':
    print("       Note: Large models (resnet_large, transformer_large) may be slow on MPS")


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print("TEST SUMMARY")
print('='*60)
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if errors:
    print(f"\nErrors:")
    for err in errors:
        print(f"  - {err}")

if failed == 0:
    print("\n[OK] All systems ready for training!")
    print("\nSuggested commands:")
    print("  # Quick test with traditional ML:")
    print("  python3 scripts/train_eval_model.py --data ./raw/chess_quality.tsv --model random_forest")
    print("\n  # Compare all model types:")
    print("  python3 scripts/train_eval_model.py --data ./raw/chess_quality.tsv --compare ridge,random_forest,mlp_small,cnn_shallow")
    sys.exit(0)
else:
    print(f"\n[ERROR] {failed} test(s) failed. Fix issues before training.")
    sys.exit(1)
