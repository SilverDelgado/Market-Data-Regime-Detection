"""
train_hmm.py
Sandbox HMM basado en tamaños de split (train/val/test) en muestras.
- Acepta lista de features y nº de estados.
- Genera 1..N splits deslizantes por tamaños (no por fechas).
- Estandariza con media/std de TRAIN (robusto).
- Mide LL/AIC/BIC; guarda mejor modelo (según ll_val_per_obs).
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from hmmlearn.hmm import GaussianHMM


# ---------------------------
# Utilidades
# ---------------------------
def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def parse_features_arg(arg_list):
    if not arg_list:
        return []
    out = []
    for token in arg_list:
        out.extend([p.strip() for p in token.split(",") if p.strip()])
    seen = set(); ordered = []
    for f in out:
        if f not in seen:
            seen.add(f); ordered.append(f)
    return ordered

def count_params_gaussian_hmm(k, d, cov_type="full"):
    startprob = k - 1
    trans = k * (k - 1)
    means = k * d
    if cov_type == "full":
        covars = k * (d * (d + 1) // 2)
    elif cov_type == "diag":
        covars = k * d
    else:
        raise ValueError(f"covariance_type no soportado: {cov_type}")
    return startprob + trans + means + covars

def standardize_with_train(df, cols, train_idx):
    """Devuelve X_std y (mu, sigma) de TRAIN para estandarizar val/test."""
    mu = df.iloc[train_idx, :][cols].mean(axis=0)
    sigma = df.iloc[train_idx, :][cols].std(axis=0).replace(0.0, 1.0)
    X = (df[cols] - mu) / sigma
    return X.values, mu.values, sigma.values

def apply_standardization(df, cols, mu, sigma):
    return ((df[cols] - mu) / sigma).values

def generate_size_splits(n_total, train_size, val_size, test_size, step_size=0, split_limit=None):
    """
    Genera splits por tamaños absolutos (en nº de filas), de forma deslizante si step_size>0.
    Retorna lista de tuplas (train_start, train_end, val_end, test_end) con índices inclusivo/exclusivo.
    """
    block = train_size + val_size + test_size
    if block > n_total:
        raise ValueError(f"Block (train+val+test={block}) > n_total={n_total}")
    starts = [0] if step_size <= 0 else list(range(0, n_total - block + 1, step_size))
    if split_limit is not None:
        starts = starts[:split_limit]
    splits = []
    for s in starts:
        train_start = s
        train_end   = s + train_size
        val_end     = train_end + val_size
        test_end    = val_end + test_size
        if test_end <= n_total:
            splits.append((train_start, train_end, val_end, test_end))
    return splits


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="HMM sandbox con splits por tamaño.")
    # rutas por defecto: data junto a este script (tu guardado actual)
    default_root = Path(__file__).resolve().parents[0]  # .../MODELOS/AI/HMM
    default_data = default_root / "data"
    default_features = default_data / "features" / "features_DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv"

    # archivos y salida
    parser.add_argument("--features-file", type=str, default=str(default_features))
    parser.add_argument("--out-root", type=str, default=str(default_data))
    parser.add_argument("--tag", type=str, default="")

    # modelo
    parser.add_argument("--features", nargs="*", default=["ret_log", "rolling_std_1h", "atr_14"])
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--cov", type=str, choices=["full", "diag"], default="full")
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # splitting por tamaños (no por fechas)
    parser.add_argument("--train-size", type=int, default=200000, help="Filas en TRAIN")
    parser.add_argument("--val-size",   type=int, default=50000,  help="Filas en VAL")
    parser.add_argument("--test-size",  type=int, default=50000,  help="Filas en TEST")
    parser.add_argument("--step-size",  type=int, default=0,      help="Paso del split deslizante (0=solo 1 split)")
    parser.add_argument("--split-limit", type=int, default=None,  help="Máx nº de splits a generar")

    args = parser.parse_args()

    out_root   = Path(args.out_root)
    models_dir = out_root / "models"
    results_dir = out_root / "results"
    ensure_dirs(models_dir, results_dir)

    features_file = Path(args.features_file)
    feature_cols = parse_features_arg(args.features)

    print(f"Cargando dataset: {features_file}")
    df = pd.read_csv(features_file, index_col="datetime", parse_dates=True)

    # validación columnas
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    n_total = len(df)
    splits = generate_size_splits(
        n_total,
        args.train_size,
        args.val_size,
        args.test_size,
        step_size=args.step_size,
        split_limit=args.split_limit
    )
    if not splits:
        raise ValueError("No se generó ningún split. Revisa train/val/test size y step_size.")

    print(f"Splits generados: {len(splits)}")
    for i, (tr_s, tr_e, va_e, te_e) in enumerate(splits, 1):
        print(f"  Split {i}: TRAIN[{tr_s}:{tr_e}) VAL[{tr_e}:{va_e}) TEST[{va_e}:{te_e})  (tam={tr_e-tr_s}/{va_e-tr_e}/{te_e-va_e})")

    # loop de experimentos por split: entrenar en TRAIN, evaluar VAL/TEST
    best_idx = None
    best_ll_val = -np.inf
    all_metrics = []

    for idx, (tr_s, tr_e, va_e, te_e) in enumerate(splits, 1):
        # Estandarización con estadísticos de TRAIN (mejor práctica)
        X_all_std, mu, sigma = standardize_with_train(df, feature_cols, slice(tr_s, tr_e))
        X_train = X_all_std[tr_s:tr_e]
        X_val   = X_all_std[tr_e:va_e] if va_e > tr_e else None
        X_test  = X_all_std[va_e:te_e] if te_e > va_e else None

        print(f"[Split {idx}] Entrenando HMM: states={args.states}, cov={args.cov}, n_iter={args.n_iter}")
        model = GaussianHMM(
            n_components=args.states,
            covariance_type=args.cov,
            n_iter=args.n_iter,
            random_state=args.seed
        ).fit(X_train)

        # Métricas
        ll_train = model.score(X_train)
        n_params = count_params_gaussian_hmm(args.states, len(feature_cols), args.cov)
        n_obs_tr = len(X_train)
        aic = 2 * n_params - 2 * ll_train
        bic = np.log(n_obs_tr) * n_params - 2 * ll_train

        metrics = {
            "split": idx,
            "states": args.states,
            "covariance_type": args.cov,
            "n_iter": args.n_iter,
            "seed": args.seed,
            "features": feature_cols,
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "test_size": int(args.test_size),
            "step_size": int(args.step_size),
            "n_params": int(n_params),
            "ll_train": float(ll_train),
            "ll_train_per_obs": float(ll_train / max(n_obs_tr,1)),
            "AIC": float(aic),
            "BIC": float(bic),
        }

        if X_val is not None and len(X_val) > 0:
            ll_val = model.score(X_val)
            metrics["ll_val"] = float(ll_val)
            metrics["ll_val_per_obs"] = float(ll_val / len(X_val))
            if ll_val > best_ll_val:
                best_ll_val = ll_val
                best_idx = (idx, model, mu, sigma)

        if X_test is not None and len(X_test) > 0:
            ll_test = model.score(X_test)
            metrics["ll_test"] = float(ll_test)
            metrics["ll_test_per_obs"] = float(ll_test / len(X_test))

        all_metrics.append(metrics)
        print(f"[Split {idx}] LL(train)={metrics['ll_train_per_obs']:.6f}  "
              f"LL(val)={metrics.get('ll_val_per_obs', float('nan')):.6f}  "
              f"LL(test)={metrics.get('ll_test_per_obs', float('nan')):.6f}")

    # Elegimos mejor split por ll_val; si no hay val, coge el primero
    if best_idx is None:
        best_idx = (1, GaussianHMM(n_components=args.states, covariance_type=args.cov, n_iter=args.n_iter, random_state=args.seed), None, None)
        # Entrena sobre el primer split si no hubo VAL:
        tr_s, tr_e, va_e, te_e = splits[0]
        X_all_std, mu, sigma = standardize_with_train(df, feature_cols, slice(tr_s, tr_e))
        best_idx = (1, best_idx[1].fit(X_all_std[tr_s:tr_e]), mu, sigma)

    split_id, best_model, mu_best, sigma_best = best_idx

    # Guardar métricas agregadas
    base_name = f"hmm_{args.states}st_{args.cov}_{'-'.join(feature_cols)}"
    if args.tag:
        base_name += f"_{args.tag}"

    models_dir = Path(args.out_root) / "models"
    results_dir = Path(args.out_root) / "results"
    ensure_dirs(models_dir, results_dir)

    metrics_file = results_dir / f"metrics_{base_name}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Guardar modelo + scaler (mu, sigma) del mejor split
    model_file = models_dir / f"{base_name}_best-split{split_id}.pkl"
    joblib.dump({
        "model": best_model,
        "features": feature_cols,
        "mu": mu_best,
        "sigma": sigma_best,
        "split_id": split_id,
        "args": vars(args)
    }, model_file)

    print(f"Modelo guardado en: {model_file}")
    print(f"Métricas guardadas en: {metrics_file}")

    # Estados sobre TODO el dataset usando el scaler de TRAIN del mejor split
    X_all = apply_standardization(df, feature_cols, mu_best, sigma_best)
    hidden_states = best_model.predict(X_all)
    df_out = df.copy()
    df_out["state"] = hidden_states

    states_csv = results_dir / f"states_{base_name}_best-split{split_id}.csv"
    df_out[["close", "state"]].to_csv(states_csv)
    print(f"Estados guardados en: {states_csv}")

    # Gráfico
    fig_file = results_dir / f"{base_name}_best-split{split_id}.png"
    plt.figure(figsize=(14, 6))
    plt.plot(df_out.index, df_out["close"], label="Close", alpha=0.85)
    plt.scatter(df_out.index, df_out["close"], c=df_out["state"], s=4, cmap="coolwarm", label="Regime")
    plt.title(f"HMM {args.states} estados | cov={args.cov} | features={', '.join(feature_cols)} | best split {split_id}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {fig_file}")


if __name__ == "__main__":
    main()
