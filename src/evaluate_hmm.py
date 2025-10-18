"""
evaluate_hmm.py
Diagnóstico de HMM entrenado:
- Carga modelo + scaler (mu, sigma) y dataset de features.
- Recalcula estados/probabilidades.
- Reporta transiciones, persistencia (duraciones), separabilidad por estado.
- Genera gráficos: prob. de estado, matriz de transición, histograma de duraciones.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def apply_standardization(df, cols, mu, sigma):
    mu = np.asarray(mu); sigma = np.asarray(sigma)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return ((df[cols].values - mu) / sigma)

def run_lengths(seq):
    """Duraciones de runs consecutivos por estado."""
    if len(seq) == 0: return []
    lengths = []
    cur_state = seq[0]; cur_len = 1
    for x in seq[1:]:
        if x == cur_state:
            cur_len += 1
        else:
            lengths.append((cur_state, cur_len))
            cur_state = x; cur_len = 1
    lengths.append((cur_state, cur_len))
    return lengths

def main():
    parser = argparse.ArgumentParser(description="Evaluación y diagnóstico de HMM entrenado.")
    default_root = Path(__file__).resolve().parents[0] / "data"
    parser.add_argument("--features-file", type=str, default=str(default_root / "features" / "features_DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv"))
    parser.add_argument("--model-file", type=str, help="Ruta al .pkl guardado por train_hmm.py (para evaluar un solo modelo)")
    parser.add_argument("--models-file", type=str, default=str(default_root / "results" / "top_k_models.json"),
                        help="Archivo JSON con rutas a modelos o lista separada por comas (para evaluar múltiples modelos)")
    parser.add_argument("--out-root", type=str, default=str(default_root), help="Carpeta base data/ para results")
    parser.add_argument("--price-col", type=str, default="Close", help="Columna de precio para los gráficos")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    results_dir = out_root / "results"
    models_dir = out_root / "models"  # Directorio de modelos
    ensure_dirs(results_dir, models_dir)

    df = pd.read_csv(args.features_file, index_col="timestamp", parse_dates=True)
    
    initial_rows = len(df)
    
    # 1. Eliminar filas con NaN en log_return
    df.dropna(subset=['log_return'], inplace=True)
    
    # 2. Eliminar fines de semana (sábado=5, domingo=6)
    if isinstance(df.index, pd.DatetimeIndex):
        is_weekday = df.index.dayofweek < 5
        df = df[is_weekday]

    final_rows = len(df)
    if initial_rows > final_rows:
        print(f"[limpieza] Filas iniciales: {initial_rows}. "
              f"Se eliminaron {initial_rows - final_rows} filas (NaNs y/o fines de semana). "
              f"Filas finales: {final_rows}.")
    
    # Evaluar un solo modelo
    if args.model_file:
        models = [args.model_file]
    # Evaluar múltiples modelos
    elif args.models_file:
        models = []
        models_file = Path(args.models_file)
        if models_file.suffix == ".json":
            with open(models_file, "r", encoding="utf-8") as f:
                models = json.load(f)
        else:
            models = args.models_file.split(",")
        # Asegurarse de que las rutas de los modelos estén en el directorio de modelos
        models = [str(models_dir / Path(m).name) for m in models]
    else:
        raise ValueError("Debes proporcionar --model-file o --models-file.")

    for model_path in models:
        model_path = Path(model_path.strip())
        if not model_path.exists():
            print(f"Modelo no encontrado: {model_path}")
            continue

        print(f"Evaluando modelo: {model_path}")
        pack = joblib.load(model_path)  # dict: model, features, mu, sigma, split_id, args
        model = pack["model"]
        feats = pack["features"]
        mu = pack["mu"]
        sigma = pack["sigma"]
        split_id = pack.get("split_id", None)

        # Estandarizar y predecir
        X_all = apply_standardization(df, feats, mu, sigma)
        states = model.predict(X_all)
        probs = model.predict_proba(X_all)  # NxK

        # Guardar resultados (sin gráficos)
        tag = args.tag.strip()
        base = model_path.stem.replace(".pkl", "")
        base_name = f"eval_{base}"
        if tag: base_name += f"_{tag}"

        csv_states = results_dir / f"{base_name}_states.csv"
        df_out = df.copy()
        for k in range(model.n_components):
            df_out[f"p_state{k}"] = probs[:, k]
        df_out["state"] = states
        df_out[[args.price_col, "state"] + [c for c in df_out.columns if c.startswith("p_state")]].to_csv(csv_states, index=True)

        # Resumen (sin gráficos)
        transmat = model.transmat_  # KxK
        startprob = getattr(model, "startprob_", None)

        rl = run_lengths(states)
        dur_by_state = {}
        for s, L in rl:
            dur_by_state.setdefault(s, []).append(L)

        sep_var = "ret_log" if "ret_log" in df.columns else feats[0]
        cond_stats = []
        for k in range(model.n_components):
            mask = states == k
            vals = df.loc[mask, sep_var].values
            if len(vals) == 0: 
                m = v = np.nan
            else:
                m, v = np.nanmean(vals), np.nanvar(vals)
            cond_stats.append({"state": int(k), "mean_"+sep_var: float(m), "var_"+sep_var: float(v), "count": int(mask.sum())})

        summary = {
            "model_file": str(model_path),
            "features_used": feats,
            "split_id": int(split_id) if split_id is not None else None,
            "transmat": transmat.tolist(),
            "startprob": startprob.tolist() if startprob is not None else None,
            "durations_by_state": {int(k): [int(x) for x in v] for k, v in dur_by_state.items()},
            "conditional_stats": cond_stats
        }
        json_file = results_dir / f"{base_name}_summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Resultados guardados para: {model_path}")

if __name__ == "__main__":
    main()
