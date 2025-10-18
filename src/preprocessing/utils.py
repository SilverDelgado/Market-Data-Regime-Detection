"""Utility helpers for loading and preprocessing market data.

Provides:
- load_csv(filepath, datetime_col='Date'/'date' etc)
- ensure_ohlcv_columns(df)
- iqr_filter(df, cols, k=1.5)
- interpolate_gaps(df, max_consecutive=3)
- indicator functions: atr, rsi, macd, bollinger_width, sma, ema
- zscore_normalize(df, cols)
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def load_csv(path: str, datetime_col: Optional[str] = None, tz: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV into a DataFrame, parse datetime if column provided or infer first column.

    Args:
        path: path to CSV
        datetime_col: name of datetime column to parse; if None, will try common names
        tz: timezone to localize index to
    Returns:
        DataFrame with DatetimeIndex
    """
    df = pd.read_csv(path)
    # guess datetime column
    if datetime_col is None:
        for candidate in ['Date', 'date', 'datetime', 'time', 'Timestamp', 'timestamp']:
            if candidate in df.columns:
                datetime_col = candidate
                break
    if datetime_col is not None and datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.set_index(datetime_col)
    else:
        # assume first column is datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.set_index(df.columns[0])

    if tz:
        df.index = df.index.tz_localize(tz)

    # convert column names to standard OHLCV if possible
    df.columns = [c.strip() for c in df.columns]
    try:
        df['volume'] = df['tick_volume']
    except KeyError:
        df['volume'] = df['volume']
    return df


def ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has columns named ['Open','High','Low','Close','Volume'] when possible."""
    mapping = {}
    cols = [c.lower() for c in df.columns]
    for i, c in enumerate(cols):
        if c in ('open', 'o'):
            mapping[df.columns[i]] = 'Open'
        elif c in ('high', 'h'):
            mapping[df.columns[i]] = 'High'
        elif c in ('low', 'l'):
            mapping[df.columns[i]] = 'Low'
        elif c in ('close', 'c', 'price'):
            mapping[df.columns[i]] = 'Close'
        elif c in ('volume', 'vol', 'v'):
            mapping[df.columns[i]] = 'Volume'
    df = df.rename(columns=mapping)
    return df


def iqr_filter(df: pd.DataFrame, cols: Optional[List[str]] = None, k: float = 1.5) -> pd.DataFrame:
    """Remove extreme outliers per column using IQR method. Returns a mask-applied df with outliers set to NaN.

    If cols is None, operate on derived numeric columns (excluding OHLC: Open, High, Low, Close).
    """
    if cols is None:
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        cols = [c for c in all_numeric if c not in ohlc_cols]
    df = df.copy()
    for c in cols:
        # skip non-existent or non-numeric columns safely
        if c not in df.columns:
            continue
        try:
            series = df[c].astype(float)
        except Exception:
            # non-numeric column, skip
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask = (series < lower) | (series > upper)
        # Use Series.mask to avoid potential index/column alignment issues
        df[c] = df[c].mask(mask, other=np.nan)
    return df


def interpolate_gaps(df: pd.DataFrame, max_consecutive: int = 3, limit_direction: str = 'both') -> pd.DataFrame:
    """Interpolate gaps but only up to max_consecutive NaNs in a row. Larger gaps remain NaN.

    Uses linear interpolation.
    """
    df = df.copy()
    # count consecutive NaNs per column and mask those > max_consecutive
    for idx, col in enumerate(df.columns):
        # use iloc to guarantee we get a Series (protects against odd column labels)
        series = df.iloc[:, idx]
        is_nan = series.isna().astype(int)
        # group consecutive True/False runs
        grp = (is_nan != is_nan.shift(1)).cumsum()
        counts = is_nan.groupby(grp).cumsum() * is_nan
        too_long = counts > max_consecutive
        # mask values belonging to too-long NaN runs
        df.iloc[:, idx] = df.iloc[:, idx].mask(too_long.astype(bool), other=np.nan)
        # now interpolate with limit=max_consecutive
        try:
            df.iloc[:, idx] = df.iloc[:, idx].interpolate(method='time', limit=max_consecutive, limit_direction=limit_direction)
        except Exception:
            # fallback to linear interpolation if time-based interpolation fails
            df.iloc[:, idx] = df.iloc[:, idx].interpolate(method='linear', limit=max_consecutive, limit_direction=limit_direction)
    return df


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute ATR from OHLC columns."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_width(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return width


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def rolling_stats(series: pd.Series, window: int = 20):
    return series.rolling(window=window, min_periods=1).mean(), series.rolling(window=window, min_periods=1).std()


def zscore_normalize(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Return a copy of df with z-score normalized columns appended using suffix '_z'.

    Tries to use sklearn.preprocessing.StandardScaler if available for robustness.
    Falls back to scipy.stats.zscore, then to a safe pandas implementation.
    Non-numeric or missing columns are skipped. Constant columns become zeros.
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols) == 0:
        return df

    # Prepare a numeric-only DataFrame for the selected columns
    numeric_df = df[cols].astype(float)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df.values)
    z_df = pd.DataFrame(scaled, index=numeric_df.index, columns=[c + '_z' for c in cols])
    
    # Concatenate z-scored columns to the original DataFrame
    # Ensure column alignment
    for col in z_df.columns:
        df[col] = z_df[col]
    return df


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std()


def garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    rolling_sum_gk = gk.rolling(window=window, min_periods=1).sum()
    gk_vol = np.sqrt(rolling_sum_gk / window)
    return gk_vol


def downside_deviation(series: pd.Series, window: int = 20, target: float = 0) -> pd.Series:
    diff = series - target
    downside = diff.where(diff < 0, 0)
    rolling_var = downside.rolling(window=window, min_periods=1).var()
    return np.sqrt(rolling_var)

def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=window, min_periods=1).mean()
    deviations = (tp - sma_tp).abs()
    mad = deviations.rolling(window=window, min_periods=1).mean()
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    di_plus = 100 * dm_plus.rolling(window=window, min_periods=1).mean() / atr
    di_minus = 100 * dm_minus.rolling(window=window, min_periods=1).mean() / atr
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = dx.rolling(window=window, min_periods=1).mean()
    return adx


def roc(series: pd.Series, window: int) -> pd.Series:
    return 100 * (series / series.shift(window) - 1)


def log_volume_zscored(df: pd.DataFrame, window: int) -> pd.Series:
    log_vol = np.log(df['Volume'] + 1)
    mean = log_vol.rolling(window=window, min_periods=1).mean()
    std = log_vol.rolling(window=window, min_periods=1).std()
    return (log_vol - mean) / std


def obv_slope(df: pd.DataFrame, window: int) -> pd.Series:
    obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    return obv - obv.shift(window)


def ad_line_slope(df: pd.DataFrame, window: int) -> pd.Series:
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfm * df['Volume']
    ad = mfv.cumsum()
    return ad - ad.shift(window)


def ma_slope(series: pd.Series, window: int) -> pd.Series:
    ma = series.rolling(window=window, min_periods=1).mean()
    return ma - ma.shift(1)


def price_vs_ma(df: pd.DataFrame, window: int) -> pd.Series:
    ma = df['Close'].rolling(window=window, min_periods=1).mean()
    return df['Close'] - ma


def ema_cross_signal(df: pd.DataFrame, window: int) -> pd.Series:
    ema_w = df['Close'].ewm(span=window, adjust=False).mean()
    sma_w = df['Close'].rolling(window=window, min_periods=1).mean()
    return np.where(ema_w > sma_w, 1, -1)
