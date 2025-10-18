"""
apply_pca.py
Orquesta el pipeline de transformación de features basado en PCA por familias.

Este script implementa un flujo de trabajo realista y sin fuga de datos:
1. Carga el dataset de indicadores crudos.
2. Divide los datos en un conjunto de entrenamiento (train set) para aprender las transformaciones.
3. Aprende los transformadores (Imputer, Scaler, PCA) para cada familia de features
   utilizando ÚNICAMENTE los datos de entrenamiento.
4. Guarda estos transformadores "ajustados" (artefactos) en un archivo .pkl para uso futuro.
5. Aplica los artefactos aprendidos para transformar el dataset COMPLETO, generando el archivo
   final con los factores PCA, listo para el entrenamiento del HMM.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import Dict, List

# --- Asegurar que el directorio raíz del proyecto esté en el path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Configuración de Familias ---
FAMILIES: Dict[str, List[str]] = {
    "volatilidad": ["rolling_std_", "atr_", "garman_klass_vol_", "downside_deviation_"],
    "momentum":    ["rsi_", "cci_", "adx_", "roc_"],
    "volumen":     ["log_volume_zscored_", "obv_slope_", "ad_line_slope_"],
    "tendencia":   ["ma_slope_", "price_vs_ma_", "ema_cross_signal_"],
}

def learn_pipeline_artefacts(df_train: pd.DataFrame, families: Dict[str, List[str]]) -> Dict:
    """Aprende los Imputers, Scalers y PCAs a partir del train set y los devuelve."""
    imputers = {}
    scalers = {}
    pcas = {}

    for fam_name, prefixes in families.items():
        feature_list = [col for col in df_train.columns if any(col.startswith(p) for p in prefixes)]
        if not feature_list:
            print(f"  - Advertencia: No se encontraron features para la familia '{fam_name}'. Se omitirá.")
            continue

        X_train_family = df_train[feature_list]

        # Aprender Imputer
        imputer = SimpleImputer(strategy="median").fit(X_train_family)
        imputers[fam_name] = imputer
        X_train_imputed = imputer.transform(X_train_family)

        # Aprender Scaler (RobustScaler es mejor para datos financieros)
        scaler = RobustScaler().fit(X_train_imputed)
        scalers[fam_name] = scaler
        
        # Aprender PCA
        X_train_scaled = scaler.transform(X_train_imputed)
        pca = PCA(n_components=1, random_state=42).fit(X_train_scaled)
        pcas[fam_name] = pca
        
        print(f"  - Familia '{fam_name}': Imputer, Scaler y PCA aprendidos.")

    return {"imputers": imputers, "scalers": scalers, "pcas": pcas, "families": families}


def apply_pipeline_transformations(df: pd.DataFrame, artefacts: Dict) -> pd.DataFrame:
    """Aplica los transformadores (artefactos) a un DataFrame completo y ordena las columnas."""
    imputers = artefacts["imputers"]
    scalers = artefacts["scalers"]
    pcas = artefacts["pcas"]
    families = artefacts["families"]
    
    # Inicializar con el mismo índice
    df_out = pd.DataFrame(index=df.index)

    # 1. Aplicar PCA y agregar columnas 'pca_familia'
    for fam_name, _ in families.items():
        if fam_name not in scalers: continue
            
        prefixes = families[fam_name]
        feature_list = [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
        if not feature_list: continue

        X_family = df[feature_list]

        # Aplicar transformadores en secuencia: .transform() SOLAMENTE
        X_imputed = imputers[fam_name].transform(X_family)
        X_scaled = scalers[fam_name].transform(X_imputed)
        factor = pcas[fam_name].transform(X_scaled)
        
        # Las columnas PCA se agregan aquí
        df_out[f'pca_{fam_name}'] = factor.flatten()

    COLUMNAS_BASE = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
    for col in COLUMNAS_BASE:
        if col in df.columns:
            # Añadimos la columna base con el nombre original
            df_out[col] = df[col]

    columnas_inicio_orden = COLUMNAS_BASE
    columnas_pca = [col for col in df_out.columns if col.startswith('pca_')]
    orden_final = [col for col in columnas_inicio_orden if col in df_out.columns] + columnas_pca
    df_out = df_out[orden_final]
    
    # df_out.dropna(inplace=True)
    
    return df_out

if __name__ == '__main__':
    # --- Orquestador Principal del Pipeline de PCA ---
    
    # Rutas basadas en la estructura del proyecto
    DATA_DIR = PROJECT_ROOT / 'data'
    INDICATORS_CSV = DATA_DIR / 'features' / 'DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01_features.csv'
    MODELS_DIR = DATA_DIR / 'models'
    ARTEFACTS_PATH = MODELS_DIR / 'pipeline_artefacts.pkl'
    FINAL_DATA_PATH = DATA_DIR / 'dataset_preprocessed' / 'DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01_preprocessed_families_pca.csv'
    
    # Configuración
    TRAIN_SET_RATIO = 0.6

    print("="*60)
    print("===== INICIANDO PIPELINE DE TRANSFORMACIÓN PCA =====")
    print("="*60)

    # --- PASO 1: Aprender y guardar los artefactos del pipeline ---
    print(f"\n[PASO 1/2] Aprendiendo transformadores del {TRAIN_SET_RATIO*100:.0f}% inicial de los datos...")
    try:
        df_indicators = pd.read_csv(INDICATORS_CSV, index_col=0, parse_dates=True)
        
        train_size = int(len(df_indicators) * TRAIN_SET_RATIO)
        df_train = df_indicators.iloc[:train_size]
        
        artefacts = learn_pipeline_artefacts(df_train, FAMILIES)
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artefacts, ARTEFACTS_PATH)
        print(f"-> Artefactos guardados en: {ARTEFACTS_PATH}")
    except Exception as e:
        print(f"[ERROR] Falló el aprendizaje de artefactos: {e}")
        sys.exit(1)

    # --- PASO 2: Aplicar los artefactos para crear el dataset final ---
    print(f"\n[PASO 2/2] Aplicando transformaciones a todo el dataset...")
    try:
        pipeline_artefacts = joblib.load(ARTEFACTS_PATH)
        df_indicators_full = pd.read_csv(INDICATORS_CSV, index_col=0, parse_dates=True)
        
        df_final = apply_pipeline_transformations(df_indicators_full, pipeline_artefacts)
        
        FINAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(FINAL_DATA_PATH, index=True) # index=True para guardar datetime
        print(f"-> Dataset final con factores PCA guardado en: {FINAL_DATA_PATH}")
        print("\nHead del dataset final:")
        print(df_final.head())
    except Exception as e:
        print(f"[ERROR] Falló la aplicación de las transformaciones: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("===== PIPELINE PCA COMPLETADO =====")
    print("="*60)