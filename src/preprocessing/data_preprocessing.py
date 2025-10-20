"""Preprocessing pipeline for Phase 1 (Preparación y Exploración de Datos).

Loads data/raw/EURUSD_15M.csv and outputs data/preprocessed/EURUSD_15M_preprocessed.csv
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# If script is executed directly, ensure project root is on sys.path so `src` is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .utils import (
    load_csv,
    ensure_ohlcv_columns,
    iqr_filter,
    interpolate_gaps,
    atr,
    rsi,
    macd,
    bollinger_width,
    sma,
    ema,
    log_returns,
    rolling_stats,
    zscore_normalize,
    rolling_std,
    garman_klass_vol,
    downside_deviation,
    cci,
    adx,
    roc,
    log_volume_zscored,
    obv_slope,
    ad_line_slope,
    ma_slope,
    price_vs_ma,
    ema_cross_signal,
)



def fib_sequence(start=6, max_val=200):
    fibs = []
    a, b = 1, 1
    while b <= max_val:
        if b >= start:
            fibs.append(b)
        a, b = b, a + b
    return fibs

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
RAW = DATA_DIR / 'raw' / 'DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv'
PRE = DATA_DIR / 'preprocessed'
PRE.mkdir(parents=True, exist_ok=True)

ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

def preprocess_eurusd_15m(raw_path: Path = RAW, out_dir: Path = PRE) -> Path:
    df = load_csv(str(raw_path))
    df = ensure_ohlcv_columns(df)

    # ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # IQR filter (set extreme numeric outliers to NaN)
    df = iqr_filter(df)

    # Interpolate small gaps up to 3 candles
    df = interpolate_gaps(df, max_consecutive=3)

    # Feature engineering
    # Price-based features from Close
    if 'Close' not in df.columns:
        # try to create Close from available columns
        if 'Price' in df.columns:
            df['Close'] = df['Price']
        else:
            # fallback to first numeric column
            numcols = df.select_dtypes(include=[np.number]).columns
            if len(numcols) == 0:
                raise ValueError('No numeric price column found in raw data')
            df['Close'] = df[numcols[0]]

    df['log_return'] = log_returns(df['Close'])

    # Additional features for Fibonacci windows
    fib_windows = fib_sequence()
    features = {}
    for w in tqdm(fib_windows, desc="Processing Fibonacci windows"):
        features[f'atr_{w}'] = atr(df, window=w)
        features[f'rsi_{w}'] = rsi(df['Close'], window=w)
        ret_mean, ret_std = rolling_stats(df['log_return'].fillna(0), window=w)
        features[f'ret_mean_{w}'] = ret_mean
        features[f'ret_std_{w}'] = ret_std
        macd_line, signal_line, hist = macd(df['Close'], fast=w, slow=2*w, signal=w//2)
        features[f'macd_{w}'] = macd_line
        features[f'macd_signal_{w}'] = signal_line
        features[f'macd_hist_{w}'] = hist
        features[f'rolling_std_{w}'] = rolling_std(df['Close'], window=w)
        features[f'garman_klass_vol_{w}'] = garman_klass_vol(df, window=w)
        features[f'downside_deviation_{w}'] = downside_deviation(df['log_return'].fillna(0), window=w)
        features[f'cci_{w}'] = cci(df, window=w)
        features[f'adx_{w}'] = adx(df, window=w)
        features[f'roc_{w}'] = roc(df['Close'], window=w)
        features[f'log_volume_zscored_{w}'] = log_volume_zscored(df, window=w)
        features[f'obv_slope_{w}'] = obv_slope(df, window=w)
        features[f'ad_line_slope_{w}'] = ad_line_slope(df, window=w)
        features[f'ma_slope_{w}'] = ma_slope(df['Close'], window=w)
        features[f'price_vs_ma_{w}'] = price_vs_ma(df, window=w)
        features[f'ema_cross_signal_{w}'] = ema_cross_signal(df, window=w)

    df = pd.concat([df, pd.DataFrame(features)], axis=1)
    
    scaler = StandardScaler()
    feature_cols_to_normalize = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ohlcv_cols]
    df[feature_cols_to_normalize] = scaler.fit_transform(df[feature_cols_to_normalize])

    df = df.reset_index().rename(columns={'index': 'time'})
    df = df.assign(id=df.index)

    # Save output
    out_path = out_dir / (raw_path.stem + '_preprocessed.csv')
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == '__main__':
    out = preprocess_eurusd_15m()
    print(f'Preprocessed file written to: {out}')
