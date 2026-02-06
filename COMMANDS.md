# Chess Position Evaluation AI - Commands Reference

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install torch torchvision scikit-learn xgboost lightgbm catboost
pip install unsloth transformers trl peft  # For LLM training
```

---

## 1. Data Collection

### Download positions from Lichess evaluation database
```bash
# Download 1M high-quality positions (depth 40+, balanced scores)
python3 scripts/download_lichess_evals.py \
    --stream \
    --limit 1000000 \
    --min-depth 40 \
    --max-score 1000 \
    --skip-mates \
    -o ./raw/chess_quality
```

### Download raw Lichess games
```bash
python3 scripts/download_lichess_db.py --month 2024-01 --output ./raw/games
```

### Extract positions from games
```bash
python3 scripts/extract_positions.py --input ./raw/games --output ./raw/chess
```

### Score positions with local Stockfish
```bash
python3 scripts/score_positions.py --input ./raw/chess --depth 20
```

### Use the scraper CLI
```bash
# Scrape chess positions from Lichess leaderboard players
python3 -m scraper.main --type chess_positions --input "blitz,bullet" --limit 10000

# Scrape images (for testing)
python3 -m scraper.main --type images --input urls.txt --limit 100 --async
```

---

## 2. Data Processing

### Consolidate position files to TSV
```bash
python3 consolidate.py
# Or use the script version:
python3 scripts/consolidate_data.py --input ./raw/chess_quality --output ./raw/chess_quality.tsv
```

### Prepare LLM training data
```bash
# Generate 50K training samples in Alpaca format
python3 prepare_llm_data.py \
    --limit 50000 \
    --min-depth 40 \
    --format alpaca \
    --split 0.05

# Other formats: sharegpt, chatml, completion
```

---

## 3. Train Neural Networks

### Train all architectures (comparison mode)
```bash
python3 scripts/train_eval_model.py \
    --data ./raw/chess_quality.tsv \
    --compare mlp_small,mlp_medium,mlp_large,cnn_shallow,cnn_medium,cnn_deep,resnet_small,resnet_medium,resnet_large
```

### Train a single model
```bash
# MLP
python3 scripts/train_eval_model.py --data ./raw/chess_quality.tsv --model mlp_large --epochs 50

# CNN
python3 scripts/train_eval_model.py --data ./raw/chess_quality.tsv --model cnn_deep --epochs 50

# ResNet
python3 scripts/train_eval_model.py --data ./raw/chess_quality.tsv --model resnet_large --epochs 50

# Transformer
python3 scripts/train_eval_model.py --data ./raw/chess_quality.tsv --model transformer_medium --epochs 50
```

### Available architectures
```
Neural Networks:
  mlp_small, mlp_medium, mlp_large
  cnn_shallow, cnn_medium, cnn_deep
  resnet_small, resnet_medium, resnet_large
  transformer_small, transformer_medium, transformer_large
  se_resnet, densenet, cnn_attention, nnue

Traditional ML:
  random_forest, xgboost, lightgbm, catboost
  gradient_boosting, ridge, lasso, elasticnet
  svr, kneighbors, decision_tree
```

---

## 4. Train LLMs

### Local training with Unsloth
```bash
# Fine-tune Phi-3.5
python3 train_llm_local.py \
    --data chess_llm_alpaca_50000_train.jsonl \
    --model phi \
    --epochs 3

# Fine-tune Llama 3.2
python3 train_llm_local.py --data chess_llm_alpaca_50000_train.jsonl --model llama

# Fine-tune Qwen 2.5
python3 train_llm_local.py --data chess_llm_alpaca_50000_train.jsonl --model qwen

# Fine-tune Mistral
python3 train_llm_local.py --data chess_llm_alpaca_50000_train.jsonl --model mistral
```

### Google Colab training
1. Upload `train_chess_llm_colab.ipynb` to Google Colab
2. Enable GPU runtime (T4 or better)
3. Upload training data or mount Google Drive
4. Run all cells

---

## 5. Inference / Testing

### Test a trained model
```bash
python3 scripts/test_training.py --model best_resnet_large.pth --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

### Run unit tests
```bash
pytest tests/unit/
```

### Run integration tests
```bash
pytest tests/integration/
```

---

## 6. Quick Start (Full Pipeline)

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -e .

# 2. Download 100K positions (smaller for testing)
python3 scripts/download_lichess_evals.py --stream --limit 100000 --min-depth 40 --max-score 1000 --skip-mates -o ./raw/chess_quality

# 3. Consolidate to TSV
python3 consolidate.py

# 4. Train a quick model
python3 scripts/train_eval_model.py --data ./raw/chess_quality.tsv --model mlp_small --epochs 10

# 5. Prepare LLM data
python3 prepare_llm_data.py --limit 10000 --format alpaca --split 0.1
```

---

## File Outputs

| Command | Output Location |
|---------|-----------------|
| download_lichess_evals.py | `raw/chess_quality/` |
| consolidate.py | `raw/chess_quality.tsv` |
| prepare_llm_data.py | `chess_llm_alpaca_*_train.jsonl`, `*_val.jsonl` |
| train_eval_model.py | `best_*.pth`, `checkpoint_*.pth` |
| train_llm_local.py | `./outputs/` (LoRA adapters) |

---

## Useful Flags

```bash
# Common flags for train_eval_model.py
--epochs N          # Number of training epochs (default: 50)
--batch-size N      # Batch size (default: 256)
--lr FLOAT          # Learning rate (default: 0.001)
--device cpu|cuda   # Force CPU or GPU
--save-dir PATH     # Where to save models

# Common flags for download_lichess_evals.py
--stream            # Stream directly (don't download full DB)
--limit N           # Max positions to collect
--min-depth N       # Minimum evaluation depth
--max-score N       # Filter extreme evaluations (centipawns)
--skip-mates        # Skip mate-in-X positions
```
