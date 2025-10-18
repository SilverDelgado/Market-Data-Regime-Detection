"""
build_feature_sets.py
Genera subsets automáticos de features (informativos + poco correlacionados) listos para experiment_hmm.py

Salida: ÚNICO JSON con:
- "feature_sets": lista de subsets seleccionados (lista de listas)
- "feature_sets_arg": cadena unida por "|" para pasar a --feature-sets
- "selected_subsets": lista de dicts con {subset, score}
"""

from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ========= Utilidades de proxies (régimen futuro) =========
def compute_future_vol(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window=window).std().shift(-window)

def build_proxies(df: pd.DataFrame, ret_col: str = "ret_log",
                  w1: int = 4, w2: int = 96) -> pd.DataFrame:
    if ret_col not in df.columns:
        raise ValueError(f"No existe la columna {ret_col} en el dataset.")
    prox = pd.DataFrame(index=df.index.copy())
    prox["rv_1h"] = compute_future_vol(df[ret_col], w1)
    prox["rv_1d"] = compute_future_vol(df[ret_col], w2)
    prox["ret_fwd_1"] = df[ret_col].shift(-1)
    prox["abs_ret_fwd_1"] = prox["ret_fwd_1"].abs()
    return prox

# ========= Scoring de features por relevancia =========
def try_import_mi():
    try:
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression
    except Exception:
        print("fallback a corr porque no se pudo importar mutual_info_regression de sklearn")
        return None

def _corr_scores_vectorized(X: pd.DataFrame, P: pd.DataFrame) -> pd.Series:
    """
    Devuelve la media de |corr| de cada feature vs todas las proxies, vectorizado.
    """
    # Alinear a numpy
    Xv = X.to_numpy(dtype=float)
    Pv = P.to_numpy(dtype=float)

    # Estandarizar columnas (evitar std=0 -> correlaciones 0)
    X_mean = np.nanmean(Xv, axis=0)
    P_mean = np.nanmean(Pv, axis=0)
    X_std = np.nanstd(Xv, axis=0, ddof=1)
    P_std = np.nanstd(Pv, axis=0, ddof=1)
    X_std[X_std == 0] = 1.0
    P_std[P_std == 0] = 1.0
    Zx = (Xv - X_mean) / X_std
    Zp = (Pv - P_mean) / P_std
    
    Zx = np.nan_to_num(Zx, nan=0.0, posinf=0.0, neginf=0.0)
    Zp = np.nan_to_num(Zp, nan=0.0, posinf=0.0, neginf=0.0)

    # Corr(X,P) = Zx^T @ Zp / (n-1)
    n = Zx.shape[0]
    if n <= 1:
        return pd.Series(0.0, index=X.columns)
    corr_mat = (Zx.T @ Zp) / (n - 1)  # shape: [n_feats, n_proxies]
    mean_abs = np.mean(np.abs(corr_mat), axis=1)
    return pd.Series(mean_abs, index=X.columns)

def score_features(df: pd.DataFrame,
                   features: List[str],
                   proxies: pd.DataFrame,
                   method: str = "auto",
                   random_state: int = 42) -> pd.Series:
    """
    Ranking de features por correlación media absoluta con proxies (vectorizado) o por MI.
    """
    df_all = pd.concat([df[features], proxies], axis=1).dropna(how="any")
    if df_all.empty:
        raise ValueError("Tras alinear features y proxies, no quedan datos (NaNs excesivos).")

    X = df_all[features].astype(float)
    P = df_all[proxies.columns].astype(float)

    if method == "auto":
        mi_fun = try_import_mi()
        method = "mi" if mi_fun is not None else "corr"

    if method == "mi":
        mi_fun = try_import_mi()
        if mi_fun is None:
            raise RuntimeError("mutual_info_regression no disponible; usa --score-method corr")

        # Secuencial (normalmente #features << #subsets; el cuello suele estar en el paso 5)
        scores = {}
        for feat in features:
            vals = []
            x = X[[feat]].values.reshape(-1, 1)
            for pcol in P.columns:
                y = P[pcol].values
                mi = mi_fun(x, y, random_state=random_state)
                vals.append(mi[0])
            scores[feat] = float(np.mean(vals))
        return pd.Series(scores).sort_values(ascending=False)

    elif method == "corr":
        s = _corr_scores_vectorized(X, P)
        return s.sort_values(ascending=False)

    else:
        raise ValueError(f"score-method desconocido: {method}")

# ========= Filtro de correlación & scoring de subsets (rápidos) =========
def _filter_subset_by_corr_fast(
    subset: List[str],
    scores: pd.Series,
    corr_abs_df: pd.DataFrame,
    corr_threshold: float
) -> List[str]:
    """
    Filtra subset usando una matriz de correlación absoluta precalculada.
    Mantiene orden por score descendente.
    """
    if len(subset) <= 1:
        return subset[:]
    ordered = sorted(subset, key=lambda f: (-scores.get(f, 0.0), f))
    kept: List[str] = []
    for f in ordered:
        if not kept:
            kept.append(f); continue
        # comprobar |rho| con todos los kept
        ok = True
        for g in kept:
            rho = corr_abs_df.loc[f, g]
            if np.isfinite(rho) and rho >= corr_threshold:
                ok = False
                break
        if ok:
            kept.append(f)
    return kept

def _subset_score_fast(
    kept: List[str],
    feat_scores: pd.Series,
    corr_abs_df: pd.DataFrame,
    diversity_weight: float
) -> float:
    base = float(np.sum([feat_scores.get(f, 0.0) for f in kept]))
    if len(kept) <= 1:
        return base
    # media de correlaciones absolutas en el subgrafo de kept (triángulo superior)
    sub = corr_abs_df.loc[kept, kept].to_numpy()
    m = sub.shape[0]
    if m <= 1:
        return base
    iu = np.triu_indices(m, k=1)
    mean_abs_corr = float(np.mean(sub[iu])) if iu[0].size > 0 else 0.0
    return base + diversity_weight * (1.0 - mean_abs_corr) * len(kept)

# ---- Worker multiproceso ----
def _filter_and_score_worker(
    subset: Tuple[str, ...],
    feat_scores_items: List[Tuple[str, float]],
    corr_abs_data: Tuple[List[str], np.ndarray],
    min_k: int,
    corr_threshold: float,
    diversity_weight: float
) -> Optional[Tuple[Tuple[str, ...], float]]:
    """
    Worker que filtra y puntúa un subset candidato.
    Recibe estructuras compactas serializables.
    """
    # Reconstruir estructuras ligeras
    feat_scores = pd.Series(dict(feat_scores_items))
    feat_names, corr_mat = corr_abs_data
    corr_abs_df = pd.DataFrame(corr_mat, index=feat_names, columns=feat_names)

    kept = _filter_subset_by_corr_fast(list(subset), feat_scores, corr_abs_df, corr_threshold)
    if len(kept) < min_k:
        return None
    kept_sorted = tuple(sorted(kept, key=lambda f: (-feat_scores.get(f, 0.0), f)))
    sc = _subset_score_fast(list(kept_sorted), feat_scores, corr_abs_df, diversity_weight)
    return kept_sorted, float(sc)

# ========= Generación de candidatos =========
def generate_candidates(top_features: List[str],
                        min_k: int,
                        max_k: int,
                        force_include: List[str] | None = None,
                        max_raw_combos: int | None = None,
                        rng: np.random.Generator | None = None) -> List[Tuple[str, ...]]:
    force_include = force_include or []
    base = [f for f in top_features if f not in force_include]
    all_combos: List[Tuple[str, ...]] = []
    for k in range(min_k, max_k + 1):
        need = max(0, k - len(force_include))
        if need < 0:
            continue
        combos = list(itertools.combinations(base, need))
        if len(combos) == 0 and len(force_include) == k:
            combos = [tuple()]
        if max_raw_combos is not None and len(combos) > max_raw_combos:
            if rng is None:
                rng = np.random.default_rng()
            idx = rng.choice(len(combos), size=max_raw_combos, replace=False)
            combos = [combos[i] for i in idx]
        for c in combos:
            subset = tuple(sorted(list(c) + list(force_include)))
            all_combos.append(subset)
    return sorted(set(all_combos))

# ========= Main =========
def main():
    ap = argparse.ArgumentParser(description="Generador automático de subsets para experiment_hmm.py (salida JSON única)")
    # Repo root: .../HMM (script is in .../HMM/src)
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "data"
    # Permite pasar un archivo CSV o un directorio; si es directorio, elegimos el CSV más reciente
    ap.add_argument("--features-file", type=str,
                    default=str(default_root / "features"))
    ap.add_argument("--score-method", type=str, choices=["auto", "mi", "corr"], default="auto")
    ap.add_argument("--train-size", type=int, default=0,
                    help="Filas para TRAIN al calcular proxies/ranking; 0 => usa 60% del dataset automáticamente")
    ap.add_argument("--ret-col", type=str, default="ret_log")
    ap.add_argument("--proxy-w1", type=int, default=4)
    ap.add_argument("--proxy-w2", type=int, default=96)
    ap.add_argument("--exclude-cols", type=str, default="open,high,low,close,volume")
    ap.add_argument("--force-include", type=str, default="ret_log")
    ap.add_argument("--top-n-features", type=int, default=12)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--corr-threshold", type=float, default=0.85)
    ap.add_argument("--max-raw-combos", type=int, default=None)
    ap.add_argument("--top-m-subsets", type=int, default=24)
    ap.add_argument("--diversity-weight", type=float, default=0.1)
    # Por defecto, guardar también en data/features
    ap.add_argument("--out-root", type=str, default=str(default_root / "features"))
    ap.add_argument("--tag", type=str, default="phase2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="-1 => usa todos los núcleos; 1 => secuencial")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Resolver archivo de features: permite directorio o archivo
    features_path = Path(args.features_file)
    if features_path.is_dir():
        # Buscar CSVs en el directorio (preferimos patrón features_*.csv, si no, cualquier .csv)
        csvs = sorted(features_path.glob("features_*.csv"))
        if not csvs:
            csvs = sorted(features_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No se encontraron CSVs en el directorio: {features_path}")
        # Elegir el más reciente por tiempo de modificación
        features_file_path = max(csvs, key=lambda p: p.stat().st_mtime)
    else:
        features_file_path = features_path

    # 1) Cargar dataset
    df = pd.read_csv(features_file_path, index_col="datetime", parse_dates=True)
    n_total = len(df)
    n_train = int(args.train_size) if args.train_size > 0 else int(round(0.60 * n_total))
    df_train = df.iloc[: n_train].copy()

    # 2) Universo de features
    exclude = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    all_feats = [c for c in df.columns if c not in exclude]
    for p in ["rv_1h", "rv_1d", "ret_fwd_1", "abs_ret_fwd_1"]:
        if p in all_feats:
            all_feats.remove(p)

    # 3) Proxies + ranking
    proxies = build_proxies(df_train, ret_col=args.ret_col, w1=args.proxy_w1, w2=args.proxy_w2)
    feat_scores = score_features(df_train, all_feats, proxies, method=args.score_method, random_state=args.seed)
    top_feats = list(feat_scores.index[: args.top_n_features])

    force_include = [c.strip() for c in args.force_include.split(",") if c.strip()]
    for f in force_include:
        if f not in top_feats and f in all_feats:
            top_feats.append(f)

    # 3.1) Matriz de correlaciones absolutas precalculada sobre el universo candidato
    #      (pairwise; NaN -> 0 para robustez en features degeneradas)
    corr_abs_df = df_train[top_feats].astype(float).corr().abs()
    corr_abs_df = corr_abs_df.fillna(0.0)
    feat_names = list(corr_abs_df.index)
    corr_abs_mat = corr_abs_df.to_numpy()

    # 4) Candidatos
    rng = np.random.default_rng(args.seed)
    raw_subsets = generate_candidates(top_features=top_feats,
                                      min_k=args.min_k,
                                      max_k=args.max_k,
                                      force_include=force_include,
                                      max_raw_combos=args.max_raw_combos,
                                      rng=rng)

    # 5) Filtro correlación + score (PARALELIZADO)
    n_jobs = os.cpu_count() if args.n_jobs is None or args.n_jobs == -1 else max(1, int(args.n_jobs))
    feat_scores_items = list(feat_scores.items())  # serializable
    corr_abs_data = (feat_names, corr_abs_mat)     # serializable y pequeño (#top_feats ~ 12)

    results: Dict[Tuple[str, ...], float] = {}

    if n_jobs == 1:
        # Secuencial
        for sub in raw_subsets:
            out = _filter_and_score_worker(
                sub,
                feat_scores_items,
                corr_abs_data,
                args.min_k,
                args.corr_threshold,
                args.diversity_weight
            )
            if out is None:
                continue
            kept, sc = out
            # Mantener el mejor score por subset (por si duplicados tras filtrado)
            if kept not in results or sc > results[kept]:
                results[kept] = sc
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _filter_and_score_worker,
                    sub,
                    feat_scores_items,
                    corr_abs_data,
                    args.min_k,
                    args.corr_threshold,
                    args.diversity_weight
                )
                for sub in raw_subsets
            ]
            for fut in as_completed(futures):
                out = fut.result()
                if out is None:
                    continue
                kept, sc = out
                if kept not in results or sc > results[kept]:
                    results[kept] = sc

    # Ordenar y truncar
    scored = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_scored = scored[: args.top_m_subsets]

    # 6) Output JSON único
    selections_list = [list(s) for s, _ in top_scored]
    feature_sets_arg = "|".join([",".join(s) for s, _ in top_scored])
    selected_subsets = [{"subset": list(s), "score": float(sc)} for s, sc in top_scored]

    tag = args.tag.strip() or "phase2"
    base = f"fs_{tag}"
    details = {
        "feature_sets": selections_list,
        "selected_subsets": selected_subsets,
        "feature_sets_arg": feature_sets_arg
    }
    with open(out_root / f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print("\nGuardado:")
    print(f" - {out_root / f'{base}.json'}")
    print(f"Data de features usada: {features_file_path}")
    print(f"Subsets generados: {len(selections_list)}  |  n_jobs={n_jobs}  |  universo={len(top_feats)} features")

if __name__ == "__main__":
    main()
