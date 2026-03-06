# StockSmart – Transformer-DRL for Inventory Optimization

StockSmart is an intelligent inventory management system that combines **transformer-based demand forecasting** with **deep reinforcement learning** to optimize replenishment decisions. The system learns optimal ordering policies by processing historical sales data, stockout events, and exogenous factors, dynamically balancing holding costs against stockout penalties.

The primary approach trains **per-category DQN agents** — one shared Deep Q-Network per product category that learns a generalized ordering policy across all products in that category. An optional **Genetic Algorithm (GA) pretraining** step can seed the DQN replay buffer with evolved (s, S) policy trajectories for faster convergence.

## Architecture

```
FreshRetailNet-50K Dataset
        │
        ▼
┌─────────────────────┐
│  Data Processing     │  Demand reconstruction, feature engineering,
│  (data_processing)   │  temporal/lag/rolling/interaction features
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Demand Forecasting  │  LSTM baseline + Temporal Fusion Transformer
│  (data_processing)   │  Quantile forecasts → state features for RL
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  RL Optimization     │  Per-category DQN agents (Stable-Baselines3)
│  (rl_optimization)   │  Optional GA pretraining (DEAP) / Gymnasium
└─────────────────────┘
```

## Dataset

**FreshRetailNet-50K** from Hugging Face — hourly sales, stockout indicators, and rich contextual features for ~50K product-store combinations.

| Split | Rows | Description |
|-------|------|-------------|
| Train | 4.5M | Historical sales with covariates |
| Eval  | 350K | Held-out evaluation partition |

Key fields: `hours_sale`, `hours_stock_status`, `discount`, `holiday_flag`, `activity_flag`, weather variables, and full category hierarchy.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| RL Framework | Stable-Baselines3 (DQN) |
| Environment API | Gymnasium |
| Deep Learning | PyTorch |
| Transformer Models | PyTorch Lightning / Hugging Face Transformers |
| Genetic Algorithm | DEAP (optional) |
| Data Processing | Pandas, NumPy |
| Forecasting | NeuralForecast (LSTM, TFT) |
| MILP Baseline | Pyomo + GLPK |
| Visualization | Matplotlib, Plotly |
| Experiment Tracking | Weights & Biases |

## Project Structure

```
RL/
├── README.md
├── requirements.txt
├── artifacts/                        # Generated models and features
│   ├── hourly_features.parquet
│   ├── rl_forecast_features.parquet
│   ├── feature_scaler.pkl
│   ├── nf_lstm/
│   ├── nf_tft/
│   └── dqn_category_<id>/           # Per-category DQN models
├── data/
│   └── data_processing.ipynb        # Data processing + demand forecasting
└── rl_optimization.ipynb            # Per-category RL training + evaluation
```

## Setup

```bash
# Clone the repo
git clone <repo-url> && cd RL

# Create a virtual environment (recommended)
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Pipeline

### 1. Data Processing & Feature Engineering (`data/data_processing.ipynb`)

- Load FreshRetailNet-50K from Hugging Face
- Reconstruct latent demand during stockout periods
- Reshape daily rows with nested hourly sequences into long-format hourly table
- Engineer temporal, lag, rolling, interaction, and hierarchical features
- Standardize features and perform chronological train/val/test split

### 2. Demand Forecasting (`data/data_processing.ipynb`)

- **LSTM baseline**: 168-hour lookback, multi-step 24-hour forecast
- **Temporal Fusion Transformer (TFT)**: probabilistic quantile forecasts with attention-based interpretability
- Evaluate on MAE, RMSE, Quantile Loss, and bias metrics
- Export point forecasts + prediction intervals as RL state features

### 3. RL Optimization (`rl_optimization.ipynb`)

Two environment types:

- **`InventoryEnv`**: single product-store environment for per-product evaluation
- **`CategoryInventoryEnv`**: multi-product environment that randomly samples a product-store from the category each episode, training the agent on diverse demand patterns

Per-category DQN training loop:

1. Select the top 5 product categories by number of product-store combinations
2. For each category, build a `CategoryInventoryEnv` with all its product-stores
3. Train a DQN agent (MlpPolicy, [256, 256], epsilon-greedy) on the category environment
4. Optionally seed the replay buffer with GA-evolved (s, S) policy trajectories (`USE_GA_PRETRAINING = True`)
5. Evaluate against (s, S), EOQ, and random baselines on the test split
6. Visualize inventory trajectories and cumulative cost per category

State vector for `CategoryInventoryEnv`:
```
[product_index_normalized, on_hand_inventory, incoming_shipments...,
 demand_forecast..., price, discount, stockout_history...]
```

Reward: negative total cost = -(holding + stockout penalty + ordering costs)

## Configuration

Key flags in `rl_optimization.ipynb`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_GA_PRETRAINING` | `False` | Enable GA (s,S) evolution + replay seeding |
| `TOTAL_TIMESTEPS` | `20,000` | DQN training steps per category |
| `N_CATEGORIES` | `5` | Number of product categories to train |
| `EPISODE_LENGTH` | `365` | Days per episode |

## Evaluation & Experiment Tracking

All experiments can be logged with **Weights & Biases**:

- Per-category cost comparison (DQN vs baselines)
- Service level (% periods without stockout)
- Inventory trajectory comparison plots
- Hyperparameter sweeps and ablation studies (GA vs no-GA)

## License

MIT
