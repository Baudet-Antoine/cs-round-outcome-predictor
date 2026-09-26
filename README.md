# CS2 Round Outcome Predictor

Neural network classifier predicting Counter-Strike 2 professional round outcomes from team economy data.

## Dataset

**sneakyzero/cs2-pro-round-economy** — 42,009 professional rounds  
- Player-level features: cash, equipment value, armor, helmet, defuser status at freeze-end
- Aggregated to team level (T/CT per round)
- 10 features after normalization
- Balanced binary target: 51.4% CT wins, 48.6% T wins

## Model Performance

| Model | Test Accuracy | ROC-AUC |
|-------|--------------|---------|
| MLP (baseline) | 64.27% | 0.6399 |
| MLP + engineered features | 64.27% | 0.6397 |
| XGBoost (100 trees, depth 6) | 63.31% | 0.6316 |

**Best model:** MLP baseline (3-layer, 64→32→1, dropout 0.3)

## Architecture

```python
class RoundOutcomePredictor(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.mlp(x)
```

Training: BCEWithLogitsLoss, Adam (lr=1e-3), 20 epochs

## Features (10)

| Index | Feature | Side |
|-------|---------|------|
| 0 | Total cash | T |
| 1 | Equipment value | T |
| 2 | Helmet count | T |
| 3 | Defuser count | T |
| 4 | Armor value | T |
| 5 | Total cash | CT |
| 6 | Equipment value | CT |
| 7 | Helmet count | CT |
| 8 | Defuser count | CT |
| 9 | Armor value | CT |

## Findings

**Economy alone plateaus at ~64-65% accuracy.** Remaining variance attributable to:
- Tactical execution and team coordination
- Individual player skill
- Positional randomness
- Clutch mechanics

Tick-level data (player positions, utility usage, damage events) required to exceed ceiling.

## Usage

```python
import torch
import pickle

# Load model
model = RoundOutcomePredictor(input_size=10)
model.load_state_dict(torch.load('round_outcome_model.pt'))
model.eval()

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict
# Input: [t_cash, t_equip, t_helmets, t_defuser, t_armor, ct_cash, ct_equip, ct_helmets, ct_defuser, ct_armor]
features = scaler.transform([[...]])
logits = model(torch.from_numpy(features).float())
prob_ct_win = torch.sigmoid(logits).item()
```

## Files

- `round_outcome_model.pt` — MLP weights
- `scaler.pkl` — StandardScaler (fit on training set)
- `model_comparison.csv` — Benchmark table
- `round_outcome_notebook.ipynb` — Full pipeline (data loading, engineering, training, evaluation)

## Requirements

```
torch>=2.0.0
scikit-learn>=1.0.0
xgboost>=1.5.0
pandas>=1.3.0
datasets>=2.0.0
huggingface-hub>=0.10.0
```

## Training

On Google Colab (free tier):
```bash
pip install torch scikit-learn xgboost pandas datasets huggingface-hub
python train.py  # or notebook cell by cell
```

RAM: ~13GB (Colab free limit), train/test 80/20 stratified split, `random_state=42`

## Conclusion

Economy prediction is calibrated and reproducible at ~64% accuracy. Advancing beyond requires structural data unavailable in open datasets. Next iteration depends on tick-level tape access or feature engineering from external sources (team ratings, player MMR, historical H2H).

---

**Author:** Antoine Baudet  
**License:** MIT  
**Dataset:** [sneakyzero/cs2-pro-round-economy](https://huggingface.co/datasets/sneakyzero/cs2-pro-round-economy)
