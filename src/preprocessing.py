
import pandas as pd
import numpy as np
import ta  # librería de technical analysis
from pathlib import Path
from scipy.stats.mstats import winsorize

# -----------------------------
# 1. Función para cargar dataset
# -----------------------------
def load_mt_dataset(filename: str, data_path: str) -> pd.DataFrame:
    filepath = Path(data_path) / filename
    df = pd.read_csv(filepath)
    # MetaQuotes suele exportar columnas como: <Date, Time, Open, High, Low, Close, Volume>
    df.columns = [c.lower() for c in df.columns]
    if 'time' in df.columns and 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + " " + df['time'])
        df.drop(['date', 'time'], axis=1, inplace=True)
    elif 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'])
        df.drop(['time'], axis=1, inplace=True)

    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

def load_dk_dataset(filename: str, data_path: str) -> pd.DataFrame:
    filepath = Path(data_path) / filename
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.drop(['timestamp'], axis=1, inplace=True)
    elif 'local time' in df.columns:
        df['datetime'] = pd.to_datetime(df['local time'])
        df.drop(['local time'], axis=1, inplace=True)
    else:
        raise ValueError(f"No encuentro columnas timestamp en {filename}. Columnas: {df.columns}")

    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

# ------------------------------------
# 2. Función para generar nuevos features
# ------------------------------------
def generate_features(df: pd.DataFrame, 
                      normalize: bool = True,
                      winsorize_limits: tuple = None,
                      features_config: dict = None) -> pd.DataFrame:
    """
    Genera features para HMM/HSMM a partir de OHLCV.
    
    Args:
        df: DataFrame con columnas ['open','high','low','close','volume'] + index datetime
        normalize: aplica z-score a todas las features
        winsorize_limits: tuple (lower, upper) para winsorizar (ej: (0.01,0.01)) o None
        features_config: dict con flags para activar/desactivar features
    
    Returns:
        DataFrame con features procesadas y normalizadas
    """
    df = df.copy()

    # Config por defecto
    if features_config is None:
        features_config = {
            "returns": True,
            "range": True,
            "volume": True,
            "indicators": True,
            "volatility": True,
            "time": True,
            "extremes": True
        }

    # --- Retornos ---
    if features_config["returns"]:
        df['ret_log'] = np.log(df['close'] / df['close'].shift(1))
        # opcional: simple return
        # df['ret_simple'] = df['close'].pct_change()

    # --- Rangos ---
    if features_config["range"]:
        df['range_hl'] = df['high'] - df['low']
        df['range_co'] = df['close'] - df['open']

    # --- Volumen ---
    if features_config["volume"]:
        df['log_volume'] = np.log1p(df['volume'])

    # --- Indicadores técnicos ---
    if features_config["indicators"]:
        # RSI
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        # MACD
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        # ADX
        df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        # MA relativa
        ma20 = df['close'].rolling(20).mean()
        df['ma20_rel'] = (df['close'] - ma20) / ma20
        # MFI
        mfi = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'],
                                     volume=df['volume'], window=14)
        df['mfi_14'] = mfi.money_flow_index()

    # --- Volatilidad ---
    if features_config["volatility"]:
        df['atr_14'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        df['rolling_std_1h'] = df['ret_log'].rolling(window=4).std()
        df['rolling_std_1d'] = df['ret_log'].rolling(window=96).std()

    # --- Hora/día ---
    if features_config["time"]:
        hours = df.index.hour
        dow = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # --- Distancia a extremos recientes ---
    if features_config["extremes"]:
        df['dist_max20'] = (df['close'] - df['close'].rolling(20).max()) / df['close'].rolling(20).max()
        df['dist_min20'] = (df['close'] - df['close'].rolling(20).min()) / df['close'].rolling(20).min()

    # --- Winsorización opcional ---
    if winsorize_limits is not None:
        for col in df.columns:
            if col not in ['open','high','low','close','volume']:
                df[col] = winsorize(df[col], limits=winsorize_limits)

    # --- Normalización ---
    if normalize:
        feature_cols = [c for c in df.columns if c not in ['open','high','low','close','volume']]
        df[feature_cols] = df[feature_cols].apply(lambda x: (x - x.mean()) / x.std())

    df.dropna(inplace=True)
    return df

# -----------------------------
# Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    data_path = r"data\dataset_raw"
    filename = "DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv"

    # df = load_mt_dataset(filename, data_path)
    df = load_dk_dataset(filename, data_path)
    print("Raw head:")
    print(df.head())

    df_features = generate_features(df,normalize=True,winsorize_limits=(0.01,0.01)) 
    print("\nPreprocessed head:")
    print(df_features.head())

    out_path = Path("data") / "features"
    out_path.mkdir(parents=True, exist_ok=True)

    out_file = out_path / f"features_{Path(filename).stem}.csv"
    df_features.to_csv(out_file)

    print(f"\nArchivo guardado en: {out_file}")