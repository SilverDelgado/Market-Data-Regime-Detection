# 📊 Market Data Regime Detection 


## 

Project for <b>detecting market regimes using machine learning models</b> (mainly HMM but will be expanded). This repository includes scripts for preparing features, training, running experiment grids, evaluating models, and visualizing states on market data (EURUSD 15M or else).

<div align="center">
<img src="data\img\image.png" alt="Overview" width="800"/>
</div>

## 

Analyzes market data to classify the current state into different regimes (e.g., bullish, bearish, ranging). The main workflow in this repository allows you to:

- Build feature sets from a CSV of prices.
- Train HMMs with splits by size (sliding train/val/test).
- Run parallelizable grids on combinations of features / number of states / covariance type.
- Evaluate models and export states and probabilities in CSV/JSON.
- Visualize states over the price series with a Streamlit app (`app.py`).

<div align="center">
<img src="data\img\image2.png" alt="Stados" width="600"/>
</div>

<div align="center">
<img src="data\img\image3.png" alt="Stados" width="600"/>
</div>



## Main structure

- `app.py` — Viewer for HMM evaluation results. 
- `src/`:
    - `train_hmm.py` — Trains HMMs using splits by size. Saves models in `src/data/models/` and results in `src/data/results/`.
    - `experiment_hmm.py` — Runs experimental grids (parallelizable with processes or threads) on sets of features, states, and covariances.
    - `evaluate_hmm.py` — Recalculates states/probabilities for saved models and generates summaries (transitions, durations, conditional statistics).
    - `preprocessing.py`, `build_feature_sets.py` — (support scripts) to generate and validate CSVs of features used by experiments.
- `data/` — Data:
    - default dataset: `dataset_raw/DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv`
    - feature CSVs and JSONs with subsets: `features/`
    - training and evaluation outputs: `models/`, `results/`

## Authors

- **SilverDelgado** — [GitHub](https://github.com/SilverDelgado)
- **Sauvageduck24** — [GitHub](https://github.com/Sauvageduck24)

---

<div align="center">

**⭐ If you like this project, star would be appreciated hehe! ⭐**
<div align="center">
<img src="src/data/market master logo.jpg" alt="Market Master Logo" width="800"/>
</div>
*See you in the markets❤️*

</div>
