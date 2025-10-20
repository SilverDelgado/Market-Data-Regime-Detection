"""
preprocessing_pipeline.py
Orquesta el pipeline completo de preprocesamiento de datos para los modelos HMM.

Este script es el punto de entrada principal para la preparación de datos y realiza 3 pasos:
1.  Genera un dataset intermedio con todos los indicadores técnicos a partir de los datos crudos.
2.  Aprende los transformadores (Scalers y PCA) a partir de un subconjunto de entrenamiento (train set)
    y guarda estos "artefactos" en un archivo .pkl para su uso posterior.
3.  Aplica los artefactos aprendidos a todo el dataset de indicadores para generar el archivo final
    con los factores PCA, listo para el entrenamiento del modelo.

Este enfoque garantiza un preprocesamiento realista y sin fuga de datos.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_preprocessing import preprocess_eurusd_15m
from src.preprocessing.apply_pca import learn_pipeline_artefacts, apply_pipeline_transformations, FAMILIES

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_PATH = DATA_DIR / 'dataset_raw' / 'DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv'
FEATURES_DATA_PATH = DATA_DIR / 'features' / (RAW_DATA_PATH.stem + '_features.csv')
FINAL_DATA_PATH = DATA_DIR / 'dataset_preprocessed' / (RAW_DATA_PATH.stem + '_preprocessed_families_pca.csv')
MODELS_DIR = DATA_DIR / 'models'
ARTEFACTS_PATH = MODELS_DIR / 'pipeline_artefacts.pkl'

TRAIN_SET_RATIO = 0.6  # 60% de los datos para aprender los transformadores

def main():
    """Ejecuta el pipeline de preprocesamiento de principio a fin."""
    
    FEATURES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================================")
    print("===== INICIANDO PIPELINE DE PREPROCESAMIENTO HMM =====")
    print("======================================================")

    print(f"\n[PASO 1/3] Generando dataset de indicadores desde: {RAW_DATA_PATH.name}")
    try:
        indicator_file_path = preprocess_eurusd_15m(raw_path=RAW_DATA_PATH, out_dir=FEATURES_DATA_PATH.parent)
        print(f"-> Dataset de indicadores guardado en: {indicator_file_path}")
    except Exception as e:
        print(f"[ERROR] Falló la generación de indicadores: {e}")
        return

    print(f"\n[PASO 2/3] Aprendiendo Scalers y PCA del {TRAIN_SET_RATIO*100:.0f}% inicial de los datos (train set)...")
    try:
        df_indicators = pd.read_csv(indicator_file_path)
        
        train_size = int(len(df_indicators) * TRAIN_SET_RATIO)
        df_train = df_indicators.iloc[:train_size]
        
        artefacts = learn_pipeline_artefacts(df_train, FAMILIES)
        
        joblib.dump(artefacts, ARTEFACTS_PATH)
        print(f"-> Artefactos de preprocesamiento (Scalers, PCAs) guardados en: {ARTEFACTS_PATH}")
    except Exception as e:
        print(f"[ERROR] Falló el aprendizaje de artefactos: {e}")
        return


    print(f"\n[PASO 3/3] Aplicando transformaciones a todo el dataset para crear el archivo final...")
    try:
        
        pipeline_artefacts = joblib.load(ARTEFACTS_PATH)
        
        df_final = apply_pipeline_transformations(df_indicators, pipeline_artefacts)
        
        df_final.to_csv(FINAL_DATA_PATH, index=True) #TRUE PA GUARDAR DATETIME
        print(f"-> Dataset final con factores PCA guardado en: {FINAL_DATA_PATH}")
        print("\nHead del dataset final:")
        print(df_final.head())
    except Exception as e:
        print(f"[ERROR] Falló la aplicación de las transformaciones: {e}")
        return

    print("\n======================================================")
    print("===== PIPELINE DE PREPROCESAMIENTO COMPLETADO =====")
    print("======================================================")


if __name__ == "__main__":
    main()
