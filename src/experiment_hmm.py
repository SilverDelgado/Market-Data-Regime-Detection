"""
experiment_hmm.py 
- Paraleliza el grid (feature_sets × states × cov)
- Cada worker guarda el mejor modelo de su configuración (si --save-models).
- Evita transferir modelos por IPC (solo devuelve métricas).
"""

from pathlib import Path
import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# ===========================
# Globals para workers (procesos)
# ===========================
_DF = None
_DF_READY = False

def _init_worker(features_file: str):
    """
    Inicializa el DataFrame en cada proceso para no re-enviarlo por IPC.
    Se ejecuta una vez por worker.
    """
    global _DF, _DF_READY
    _DF = pd.read_csv(features_file, index_col="datetime", parse_dates=True)
    _DF_READY = True

# ---------------------------
# Utilidades
# ---------------------------
def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def parse_list_arg(arg: str):
    items = [x.strip() for x in arg.split(",") if x.strip()]
    out = []
    for it in items:
        try:
            out.append(int(it))
        except ValueError:
            out.append(it)
    return out

def parse_feature_sets(arg: str):
    """
    Admite:
    - Cadena 'a,b|c,d'
    - JSON con claves: feature_sets / selected_subsets / feature_sets_arg
    - Lista de listas en JSON
    """
    p = Path(arg)
    if p.suffix.lower() == ".json" and p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "feature_sets" in data and isinstance(data["feature_sets"], list):
                return [list(dict.fromkeys(map(str.strip, fs))) for fs in data["feature_sets"]]
            if "selected_subsets" in data and isinstance(data["selected_subsets"], list):
                return [list(dict.fromkeys(map(str.strip, d["subset"]))) for d in data["selected_subsets"] if "subset" in d]
            if "feature_sets_arg" in data and isinstance(data["feature_sets_arg"], str):
                arg = data["feature_sets_arg"]
            else:
                raise ValueError("JSON de feature-sets no tiene una clave reconocida ('feature_sets', 'selected_subsets' o 'feature_sets_arg').")
        elif isinstance(data, list) and all(isinstance(x, list) for x in data):
            return [list(dict.fromkeys(map(str.strip, fs))) for fs in data]
        else:
            raise ValueError("Formato JSON de feature-sets no soportado.")

    groups = [g.strip() for g in arg.split("|") if g.strip()]
    sets_ = []
    for g in groups:
        feats = [f.strip() for f in g.split(",") if f.strip()]
        seen = set(); ordered = []
        for f in feats:
            if f not in seen:
                seen.add(f); ordered.append(f)
        sets_.append(ordered)
    return sets_

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

def standardize_with_train(df, cols, train_slice):
    mu = df.iloc[train_slice, :][cols].mean(axis=0)
    sigma = df.iloc[train_slice, :][cols].std(axis=0).replace(0.0, 1.0)
    X_all_std = (df[cols] - mu) / sigma
    return X_all_std.values, mu.values, sigma.values

def generate_size_splits(n_total, train_size, val_size, test_size, step_size=0, split_limit=None):
    block = train_size + val_size + test_size
    if block > n_total:
        raise ValueError(f"Block (train+val+test={block}) > n_total={n_total}")
    starts = [0] if step_size <= 0 else list(range(0, n_total - block + 1, step_size))
    if split_limit is not None:
        starts = starts[:split_limit]
    splits = []
    for s in starts:
        tr_s = s
        tr_e = s + train_size
        va_e = tr_e + val_size
        te_e = va_e + test_size
        if te_e <= n_total:
            splits.append((tr_s, tr_e, va_e, te_e))
    return splits

# ---------------------------
# Experimento por configuración
# ---------------------------
def run_config(df, feature_cols, n_states, cov, n_iter, seed,
               train_size, val_size, test_size, step_size, split_limit):
    n_total = len(df)
    splits = generate_size_splits(n_total, train_size, val_size, test_size, step_size, split_limit)
    per_split_metrics = []
    best = {"ll_val": -np.inf, "model": None, "mu": None, "sigma": None, "split_id": None}

    for i, (tr_s, tr_e, va_e, te_e) in enumerate(splits, 1):
        X_all_std, mu, sigma = standardize_with_train(df, feature_cols, slice(tr_s, tr_e))
        X_train = X_all_std[tr_s:tr_e]
        X_val   = X_all_std[tr_e:va_e] if va_e > tr_e else None
        X_test  = X_all_std[va_e:te_e] if te_e > va_e else None

        model = GaussianHMM(
            n_components=n_states,
            covariance_type=cov,
            n_iter=n_iter,
            random_state=seed
        ).fit(X_train)

        ll_train = model.score(X_train)
        n_params = count_params_gaussian_hmm(n_states, len(feature_cols), cov)
        aic = 2 * n_params - 2 * ll_train
        bic = np.log(max(len(X_train), 1)) * n_params - 2 * ll_train

        row = {
            "split": i,
            "states": n_states,
            "cov": cov,
            "n_iter": n_iter,
            "seed": seed,
            "features": feature_cols,
            "n_params": int(n_params),
            "ll_train": float(ll_train),
            "ll_train_per_obs": float(ll_train / max(len(X_train), 1)),
            "AIC": float(aic),
            "BIC": float(bic)
        }

        if X_val is not None and len(X_val) > 0:
            ll_val = model.score(X_val)
            row["ll_val"] = float(ll_val)
            row["ll_val_per_obs"] = float(ll_val / len(X_val))
            if ll_val > best["ll_val"]:
                best = {"ll_val": ll_val, "model": model, "mu": mu, "sigma": sigma, "split_id": i}

        if X_test is not None and len(X_test) > 0:
            ll_test = model.score(X_test)
            row["ll_test"] = float(ll_test)
            row["ll_test_per_obs"] = float(ll_test / len(X_test))

        per_split_metrics.append(row)

    dfm = pd.DataFrame(per_split_metrics)
    agg = {
        "states": n_states,
        "cov": cov,
        "features": feature_cols,
        "splits": len(per_split_metrics),
        "ll_train_per_obs_mean": float(dfm["ll_train_per_obs"].mean()),
        "ll_train_per_obs_median": float(dfm["ll_train_per_obs"].median()),
        "AIC_mean": float(dfm["AIC"].mean()),
        "BIC_mean": float(dfm["BIC"].mean()),
    }
    if "ll_val_per_obs" in dfm.columns:
        agg["ll_val_per_obs_mean"] = float(dfm["ll_val_per_obs"].mean())
        agg["ll_val_per_obs_median"] = float(dfm["ll_val_per_obs"].median())
    if "ll_test_per_obs" in dfm.columns:
        agg["ll_test_per_obs_mean"] = float(dfm["ll_test_per_obs"].mean())
        agg["ll_test_per_obs_median"] = float(dfm["ll_test_per_obs"].median())

    return per_split_metrics, agg, best

# ---------------------------
# Worker de una configuración
# ---------------------------
def _run_single_config_worker(
    fs, k, cov, args_dict, models_dir_str, use_global_df: bool
):
    """
    Ejecuta una configuración y (opcional) guarda su mejor modelo.
    Devuelve: {split_rows, agg_row, saved_model, error}
    """
    try:
        if use_global_df:
            if not _DF_READY:
                raise RuntimeError("DF global no inicializado en worker.")
            df = _DF
        else:
            df = args_dict["df"]  # threads o modo secuencial

        split_rows, agg_row, best = run_config(
            df=df,
            feature_cols=fs,
            n_states=int(k),
            cov=str(cov),
            n_iter=int(args_dict["n_iter"]),
            seed=int(args_dict["seed"]),
            train_size=int(args_dict["train_size"]),
            val_size=int(args_dict["val_size"]),
            test_size=int(args_dict["test_size"]),
            step_size=int(args_dict["step_size"]),
            split_limit=args_dict["split_limit"]
        )

        cfg_id = f"st{k}_{cov}_{'-'.join(fs)}"
        for r in split_rows:
            r["config_id"] = cfg_id
        agg_row["config_id"] = cfg_id

        saved_model_path = None
        if args_dict["save_models"] and best["model"] is not None:
            pack = {
                "model": best["model"],
                "features": fs,
                "mu": best["mu"],
                "sigma": best["sigma"],
                "split_id": best["split_id"],
                "args": args_dict["raw_args"]
            }
            models_dir = Path(models_dir_str)
            model_file = models_dir / f"grid_best_{cfg_id}.pkl"
            joblib.dump(pack, model_file)
            saved_model_path = str(model_file)

        return {"split_rows": split_rows, "agg_row": agg_row, "saved_model": saved_model_path, "error": None}
    except Exception as e:
        return {"split_rows": [], "agg_row": None, "saved_model": None, "error": f"{type(e).__name__}: {e}"}

# ---------------------------
# Main (grid)
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Grid de HMM (features × states × cov) con splits por tamaños (paralelizable).")
    default_root = Path(__file__).resolve().parents[0] / "data"
    default_features = Path(__file__).resolve().parents[1] / "data" / "features" / "features_DUKASCOPY_EURUSD_15_2000-01-01_2025-01-01.csv"

    parser.add_argument("--features-file", type=str, default=str(default_features))
    parser.add_argument("--out-root", type=str, default=str(default_root))
    parser.add_argument("--tag", type=str, default="")

    parser.add_argument("--feature-sets", type=str, required=True,
                        help='Conjuntos de features separados por "|" o JSON.')
    parser.add_argument("--states-list", type=str, default="2,3,4,5")
    parser.add_argument("--cov-list", type=str, default="full,diag")
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-size", type=int, default=200000)
    parser.add_argument("--val-size",   type=int, default=50000)
    parser.add_argument("--test-size",  type=int, default=50000)
    parser.add_argument("--step-size",  type=int, default=0)
    parser.add_argument("--split-limit", type=int, default=None)

    parser.add_argument("--save-models", action="store_true", default=True, help="Guardar mejor modelo por configuración.")
    parser.add_argument("--keep-top-k", type=int, default=8, help="Solo informar top-K por ll_val_per_obs_mean.")

    # === NUEVO: paralelización ===
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Número de workers. -1 = todos los núcleos.")
    parser.add_argument("--backend", choices=["processes", "threads"], default="processes",
                        help="Backend de paralelización.")
    parser.add_argument("--blas-threads", type=int, default=1,
                        help="Limitar hilos internos de BLAS por worker (OMP/MKL).")

    args = parser.parse_args()

    # Limitar hilos de BLAS (mejor antes de cargar NumPy; aun así ayuda a evitar sobre-suscripción)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.blas_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.blas_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.blas_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.blas_threads))

    out_root    = Path(args.out_root)
    results_dir = out_root / "results"
    models_dir  = out_root / "models"
    ensure_dirs(results_dir, models_dir)

    feature_sets = parse_feature_sets(args.feature_sets)
    states_list  = parse_list_arg(args.states_list)
    cov_list     = parse_list_arg(args.cov_list)

    # Validación de columnas (sin cargar todo el CSV si vamos con procesos)
    if args.backend == "processes" and args.n_jobs != 1:
        cols_df = pd.read_csv(args.features_file, nrows=0)  # solo cabecera
        all_cols = set(cols_df.columns) - {"datetime"}
        # no necesitamos df aquí; cada worker cargará el CSV
        df_main = None
    else:
        df_main = pd.read_csv(args.features_file, index_col="datetime", parse_dates=True)
        all_cols = set(df_main.columns)

    for fs in feature_sets:
        miss = [c for c in fs if c not in all_cols]
        if miss:
            raise ValueError(f"Faltan columnas en el CSV para {fs}: {miss}")

    # Construir lista de tareas (combinaciones)
    tasks = []
    for fs in feature_sets:
        for k in states_list:
            for cov in cov_list:
                tasks.append((fs, int(k), str(cov)))

    # Recolectores
    all_split_rows = []
    all_cfg_rows = []
    saved_models = []
    errors = []

    # Empaquetar args ligeros para workers
    args_dict = {
        "n_iter": int(args.n_iter),
        "seed": int(args.seed),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "step_size": int(args.step_size),
        "split_limit": args.split_limit,
        "save_models": bool(args.save_models),
        "raw_args": vars(args)
    }
    if df_main is not None:
        args_dict["df"] = df_main  # compartido entre hilos (no se copia)

    # Ejecutar
    n_jobs = os.cpu_count() if int(args.n_jobs) == -1 else max(1, int(args.n_jobs))
    use_processes = (args.backend == "processes" and n_jobs != 1)

    if use_processes:
        print(f"[MP] Ejecutando con {n_jobs} procesos...")
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(args.features_file,)
        ) as ex:
            futures = []
            for fs, k, cov in tasks:
                futures.append(
                    ex.submit(_run_single_config_worker, fs, k, cov, args_dict, str(models_dir), True)
                )
            for fut in as_completed(futures):
                res = fut.result()
                # Progreso por configuración
                cid = res["agg_row"]["config_id"] if (res and res.get("agg_row")) else "?"

                if res["error"]:
                    print(f"✗ {cid}  ERROR: {res['error']}")
                else:
                    saved = "sí" if res["saved_model"] else "no"
                    print(f"✓ {cid}  OK  (model guardado: {saved})")

                if res["error"]:
                    errors.append(res["error"])
                else:
                    all_split_rows.extend(res["split_rows"])
                    all_cfg_rows.append(res["agg_row"])
                    if res["saved_model"]:
                        saved_models.append(res["saved_model"])
    elif n_jobs == 1:
        print("[SEQ] Ejecutando en modo secuencial...")
        for fs, k, cov in tasks:
            res = _run_single_config_worker(fs, k, cov, args_dict, str(models_dir), use_global_df=False)
            if res["error"]:
                errors.append(res["error"])
            else:
                all_split_rows.extend(res["split_rows"])
                all_cfg_rows.append(res["agg_row"])
                if res["saved_model"]:
                    saved_models.append(res["saved_model"])
    else:
        print(f"[TH] Ejecutando con {n_jobs} hilos...")
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = []
            for fs, k, cov in tasks:
                futures.append(
                    ex.submit(_run_single_config_worker, fs, k, cov, args_dict, str(models_dir), False)
                )
            for fut in as_completed(futures):
                res = fut.result()
                # Progreso por configuración
                cid = res["agg_row"]["config_id"] if (res and res.get("agg_row")) else "?"

                if res["error"]:
                    print(f"✗ {cid}  ERROR: {res['error']}")
                else:
                    saved = "sí" if res["saved_model"] else "no"
                    print(f"✓ {cid}  OK  (model guardado: {saved})")

                if res["error"]:
                    errors.append(res["error"])
                else:
                    all_split_rows.extend(res["split_rows"])
                    all_cfg_rows.append(res["agg_row"])
                    if res["saved_model"]:
                        saved_models.append(res["saved_model"])

    # Guardar CSVs resumen
    tag = args.tag.strip()
    base = f"grid_{tag}" if tag else "grid"

    df_splits = pd.DataFrame(all_split_rows)
    df_cfg    = pd.DataFrame(all_cfg_rows)

    splits_csv = results_dir / f"{base}_per-split.csv"
    cfg_csv    = results_dir / f"{base}_summary.csv"
    if not df_splits.empty:
        df_splits.to_csv(splits_csv, index=False)
        print(f"\nGuardado per-split: {splits_csv}")
    if not df_cfg.empty:
        df_cfg.sort_values(by=["ll_val_per_obs_mean"], ascending=False, inplace=True, na_position="last")
        df_cfg.to_csv(cfg_csv, index=False)
        print(f"Guardado summary:   {cfg_csv}")

        # Top-K (informativo)
        if args.save_models and args.keep_top_k is not None:
            top_cfgs = df_cfg.dropna(subset=["ll_val_per_obs_mean"]).head(int(args.keep_top_k))["config_id"].tolist()
            print("\nTop-K por ll_val_per_obs_mean:")
            for cid in top_cfgs:
                print(f" - {cid}")

    if errors:
        print("\n[WARN] Algunas configuraciones fallaron:")
        for e in errors:
            print(" -", e)

if __name__ == "__main__":
    # Recomendado en Windows para 'spawn'
    mp.freeze_support()
    main()
