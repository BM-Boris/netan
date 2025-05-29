# views.py ————— BuildNetworkView (Rodin API) ——————————————————————————

from __future__ import annotations

import io
import itertools, threading, uuid
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rest_framework import status
from rest_framework.views import APIView
from django.http import JsonResponse
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import graphical_lasso
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning,DataConversionWarning
import warnings, math, time, random



import rodin  # internal library

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Globals for very‑lightweight task queue (memory only)
# ──────────────────────────────────────────────────────────────────────────────
TASKS: dict[str, Dict] = {}          # {id:{status,progress,cancelled,…}}
TASKS_LOCK = threading.Lock()
JOBLIB_N_JOBS = min(4, os.cpu_count() or 1)


MAX_EDGES = 10_000  # Plotly starts lagging badly above this number
SINGLE_OMIC_GUARD = (
    "Only a single omics file uploaded. Use layerMode=\"stack\" (Entire) "
    "instead of multilayer."
)

# ═════════════════════════════════════════════════════════════════════════════
# 1  Helpers – file I/O
# ═════════════════════════════════════════════════════════════════════════════

def _detect_sep(f) -> str:
    """Auto‑detect CSV separator (comma vs tab)."""
    f.seek(0)
    line = f.readline()
    if isinstance(line, bytes):
        line = line.decode("utf‑8", "replace")
    f.seek(0)
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    raise ValueError("Cannot detect separator (no ',' or '\t' found).")


def _read_feat(f, ftype: str):
    """Read a features table, return (df, leading_cols, sample_cols)."""
    sep = _detect_sep(f)
    f.seek(0)
    df = pd.read_csv(f, sep=sep)
    f.seek(0)

    if ftype == "metabolomics":
        if df.shape[1] < 3:
            raise ValueError("Metabolomics ⇒ need ≥3 columns (mz, rt, samples …).")
        lead, samples = list(df.columns[:2]), list(df.columns[2:])
    else:
        if df.shape[1] < 2:
            raise ValueError(f"{ftype}: need ≥2 columns (id, samples …).")
        lead, samples = [df.columns[0]], list(df.columns[1:])

    return df, lead, samples


def _read_meta(f):
    sep = _detect_sep(f)
    f.seek(0)
    df = pd.read_csv(f, sep=sep)
    f.seek(0)
    if df.empty:
        raise ValueError("Meta file is empty.")
    return df, df.iloc[:, 0].tolist()


def _common_samples(feats) -> List[str]:
    """Intersect sample‑columns across all uploaded omics tables."""
    common: set[str] = set(feats[0][2])
    for _, _, cols, *_ in feats[1:]:
        common &= set(cols)
    if not common:
        raise ValueError("No shared sample columns across omics files.")
    # preserve order from the first table
    return [c for c in feats[0][2] if c in common]


def _fake_meta(samples: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": samples})


def _align_feat_meta(
    df_f, lead, samples, df_m, meta_ids
) -> Tuple[io.BytesIO, io.BytesIO]:
    """Create temporary CSV buffers aligned on common sample IDs."""
    overlap = [s for s in samples if s in meta_ids]
    if not overlap:
        raise ValueError("No sample overlap between data and metadata.")

    df_feat = df_f[lead + overlap].copy()

    df_meta = df_m.copy()
    df_meta.columns = ["sample_id"] + list(df_meta.columns[1:])
    df_meta = df_meta[df_meta["sample_id"].isin(overlap)]
    df_meta["sample_id"] = pd.Categorical(
        df_meta["sample_id"], categories=overlap, ordered=True
    )
    df_meta = df_meta.sort_values("sample_id")

    fb = io.BytesIO(df_feat.to_csv(index=False).encode())
    fb.name = "feat_tmp.csv"
    mb = io.BytesIO(df_meta.to_csv(index=False).encode())
    mb.name = "meta_tmp.csv"
    return fb, mb


def _load_rodins(files, types):
    """Return a list of rodin objects for each omics file (+ meta)."""
    meta = None
    feats = []
    for f, t in zip(files, types):
        t = t.lower()
        if t == "meta":
            if meta:
                raise ValueError("Only one meta file allowed.")
            meta = f
        else:
            feats.append((*_read_feat(f, t), t, f))

    if not feats:
        raise ValueError("No omics files found.")

    if meta:
        df_m, meta_ids = _read_meta(meta)
    else:
        df_m, meta_ids = None, []

    ordered = _common_samples(feats)
    if df_m is None:
        df_m, meta_ids = _fake_meta(ordered), ordered

    rodins = []
    ref_cols = None
    for df_f, lead, _, ftype, f_raw in feats:
        fb, mb = _align_feat_meta(df_f, lead, ordered, df_m, meta_ids)
        r = rodin.create_object_csv(
            fb,
            mb,
            ",",
            ",",
            feat_stat=None if ftype == "metabolomics" else "ann",
        )

        # preserve file metadata
        r.uns = dict(
            r.uns or {}, file_name=getattr(f_raw, "name", "file"), file_type=ftype
        )
        cols = list(r.X.columns)

        if ref_cols is None:
            ref_cols = cols
        elif cols != ref_cols:
            raise ValueError("Sample‑column mismatch between omics files.")

        rodins.append(r)

    return rodins


# ═════════════════════════════════════════════════════════════════════════════
# 2  Pre‑processing / filtering
# ═════════════════════════════════════════════════════════════════════════════

def _preproc(r, p: Dict):
    """Wrapper around r.transform() that records log‑use in r.uns."""
    if not p:
        return r
    r.transform(
        thresh=float(p.get("threshold", 1)),
        norm=None if p.get("normalization", "none") == "none" else p["normalization"],
        scale=None if p.get("scaling", "none") == "none" else p["scaling"],
        log=p.get("logTransformation", "none") == "log2",
    )
    r.uns["log"] = p.get("logTransformation", "none") == "log2"
    return r


def _filter(r, f: Dict):
    """Feature filtering based on statistical tests or fold‑change."""
    if not f or f.get("method", "none").lower() == "none":
        return r

    meth = f["method"].lower()
    thr = float(f.get("Threshold", 1))
    meta = f.get("meta", "")

    mapping = {
        "ttest": ("p_value(tt)", False, "<", r.ttest),
        "one-way anova": ("p_value(owa)", False, "<", r.oneway_anova),
        "oneway_anova": ("p_value(owa)", False, "<", r.oneway_anova),
        "pls_da": ("vips", False, ">", r.pls_da),
        "sf_lg": ("p_value(lg)", False, "<", r.sf_lg),
        "sf_lr": ("p_value(lr)", False, "<", r.sf_lr),
        "rf_class": ("imp(rf)", False, ">", r.rf_class),
        "rf_regress": ("imp(rf)", False, ">", r.rf_regress),
    }

    if meth == "fold_change":
        if r.uns.get("log") is False:
            raise ValueError("Apply log2 transformation before fold‑change.")
        key, abs_flag, op, fn = "lfc", True, ">", r.fold_change
    else:
        if meth not in mapping:
            return r
        key, abs_flag, op, fn = mapping[meth]

    fn(meta)
    feats = r.features
    cols = [c for c in feats.columns if key in c.lower()]
    if not cols:
        return r

    mask = pd.Series(True, index=feats.index)
    for c in cols:
        v = feats[c].abs() if abs_flag else feats[c]
        mask &= (v < thr) if op == "<" else (v > thr)

    return r[feats[mask]]


# ═════════════════════════════════════════════════════════════════════════════
# 3  Network builders
# ═════════════════════════════════════════════════════════════════════════════

def _df(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2  Builders with progress callbacks
# ──────────────────────────────────────────────────────────────────────────────
def _corr(df, thr, weight_flag):  # ≈ мгновенно → прогресс не нужен
    cor = df.corr("spearman")
    adj = (cor.abs() >= thr).astype(int)
    np.fill_diagonal(adj.values, 0)
    G = nx.from_pandas_adjacency(adj)
    if weight_flag == "on":
        for u, v in G.edges():
            G[u][v]["weight"] = float(abs(cor.loc[u, v]))
    return G, cor.values


# ────────────────────────────────────────────────────────────────────────────
# Fast CLR  (+ early‑abort on edge overflow)
# ────────────────────────────────────────────────────────────────────────────
def _clr(
    df, thr, weight_flag,
    *, progress=None, offset=0, span=60, cancel_cb=None,
    chunk=8, n_neighbors=2,
):
    X   = df.values.astype("float32", copy=False)
    ids = df.columns.to_list()
    p   = X.shape[1]
    MI  = np.zeros((p, p), dtype=np.float32)
    edge_count = 0           # счётчик рёбер, которые уже «известны»

    def mi_column(j):
        return mutual_info_regression(
            X, X[:, j],
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=0
        )

    for start in range(0, p, chunk):
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled by user")

        cols = range(start, min(start + chunk, p))
        res  = Parallel(n_jobs=JOBLIB_N_JOBS, prefer="threads")(
            delayed(mi_column)(j) for j in cols
        )
        MI[:, cols] = np.column_stack(res)     # shape (p, chunk)

        # — примерно оцениваем новые рёбра: только связи со старыми колонками
        for j in cols:
            edge_count += np.sum(MI[:j, j] >= thr)
            if edge_count > MAX_EDGES:
                raise RuntimeError(f"Network too dense (limit ≈ {MAX_EDGES}). Increase edgeThreshold.")

        if progress:
            done = (start + len(cols)) / p
            progress(offset + done * span * 0.5)

    # sym + CLR
    MI   = (MI + MI.T) * 0.5
    mu   = MI.mean(1, keepdims=True)
    sig  = MI.std(1, keepdims=True) + 1e-9
    S    = np.sqrt(((MI - mu) / sig) ** 2 +
                   (((MI - mu) / sig).T) ** 2)

    if progress:
        progress(offset + span * 0.9)

    # threshold → adjacency
    adj = (S >= thr).astype(np.uint8)
    np.fill_diagonal(adj, 0)
    if adj.sum() // 2 > MAX_EDGES:
        raise RuntimeError("too many edges")

    G = nx.from_pandas_adjacency(
        pd.DataFrame(adj, index=ids, columns=ids)
    )
    if weight_flag == "on":
        for u, v in G.edges():
            G[u][v]["weight"] = float(abs(S[ids.index(u), ids.index(v)]))

    if progress:
        progress(offset + span)

    return G, S


# ────────────────────────────────────────────────────────────────────────────
# Fast ExtraTrees importance  (+ early‑abort)
# ────────────────────────────────────────────────────────────────────────────


def _rf(
    df, thr, weight_flag,
    *, progress=None, offset=0, span=60, cancel_cb=None,
    chunk=4, n_estimators=80, max_depth=None,):
        
    X   = df.values.astype("float32", copy=False)
    ids = df.columns.to_list()
    p   = len(ids)
    W   = np.zeros((p, p), dtype=np.float32)
    edge_count = 0
    if max_depth==0:
        max_depth=None

    def fit_target(t):
        y  = X[:, t]
        Xo = np.delete(X, t, axis=1)
        mdl = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=1,
            max_features="sqrt"
        )
        mdl.fit(Xo, y)
        row = np.zeros(p, dtype=np.float32)
        row[np.arange(p) != t] = mdl.feature_importances_
        return t, row

    for start in range(0, p, chunk):
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled by user")

        idx  = range(start, min(start + chunk, p))
        rows = Parallel(n_jobs=JOBLIB_N_JOBS, prefer="threads")(
            delayed(fit_target)(t) for t in idx
        )
        for t, row in rows:
            W[t] = row
            edge_count += np.sum(row[:t] >= thr)
            if edge_count > MAX_EDGES:
                raise RuntimeError(f"Network too dense (limit ≈ {MAX_EDGES}). Increase edgeThreshold.")

        if progress:
            done = (start + len(idx)) / p
            progress(offset + done * span * 0.9)

    W = (W + W.T) * 0.5
    if thr is None:
        thr = np.nanpercentile(W[~np.eye(p, dtype=bool)], 95)

    adj = (W >= thr).astype(np.uint8)
    np.fill_diagonal(adj, 0)
    if adj.sum() // 2 > MAX_EDGES:
        raise RuntimeError("too many edges")

    G = nx.from_pandas_adjacency(
        pd.DataFrame(adj, index=ids, columns=ids)
    )
    if weight_flag == "on":
        for u, v in G.edges():
            G[u][v]["weight"] = float(W[ids.index(u), ids.index(v)])

    if progress:
        progress(offset + span)

    return G, W
    
# ────────────────────────────────────────────────────────────────────────────
# Graphical Lasso  (sparse inverse‑covariance)
# ────────────────────────────────────────────────────────────────────────────
def _glasso(
    df: pd.DataFrame,
    thr: float,
    weight_flag: str,
    *,
    progress=None,         # callable(float 0‑100)
    offset: float = 0.0,   # где «врезается» наш блок в общую шкалу
    span: float  = 60.0,   # ширина блока (%)
    cancel_cb=None,        # callable() → True, если пользователь отменил
    # ─ Glasso гиперпараметры + fallback
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float   = 1e-4,
    ridge_factor: float = 10.0,   # при не‑SPD усиливаем штраф × этот множитель
    max_ridge_tries: int = 8,     # макс. число таких «усиливаний»
):
    """
    Graphical Lasso с динамическим прогресс‑баром, авто‑откатом
    и ранней остановкой, если сеть получается слишком плотной.
    """

    # ─────────────────────────── константы ────────────────────────────────
    CALIB_P     = 30          # размер подматрицы для калибровки
    MIN_T_PRED  = 0.5         # ≥ 0.5 c, чтобы шкала не «дёргалась»

    if cancel_cb and cancel_cb():
        raise RuntimeError("cancelled by user")
    if progress:
        progress(offset)

    # 0) данные → float32
    X   = df.values.astype("float32", copy=False)
    ids = df.columns.to_list()
    p   = X.shape[1]

    # ── быстрая калибровка времени + прогноза числа рёбер ────────────────
    def _estimate_runtime_and_edges() -> tuple[float, int]:
        p_sub = min(p, CALIB_P)
        Xsub  = X[:, :p_sub]                    # первые CALIB_P признаков

        t0    = time.perf_counter()
        mdl   = GraphicalLasso(alpha=alpha, max_iter=max_iter // 4 or 1,
                               tol=1e-3).fit(Xsub)
        dt    = max(time.perf_counter() - t0, 1e-3)
        k     = dt / (p_sub ** 3 * (max_iter // 4 or 1))
        t_pred = max(k * p ** 3 * max_iter, MIN_T_PRED)

        # partial corr → рёбра на куске
        prec  = mdl.precision_
        d_inv = 1.0 / np.sqrt(np.diag(prec))
        Psub  = -prec * d_inv[:, None] * d_inv[None, :]
        np.fill_diagonal(Psub, 0)
        # считаем только верхний треугольник
        edges_sub = int(np.count_nonzero(np.triu(np.abs(Psub) >= thr, k=1)))
        # маштабируем на полный размер (≈ пропорционально C(p,2))
        coeff     = (p * (p - 1)) / (p_sub * (p_sub - 1))
        edges_pred = int(round(edges_sub * coeff))/15
        print(edges_pred)

        return t_pred, edges_pred

    t_pred, edges_pred = _estimate_runtime_and_edges()
    if edges_pred > MAX_EDGES:
        raise RuntimeError(
            f"Network too dense (limit ≈ {MAX_EDGES}). Increase edgeThreshold."
        )

    t0 = time.perf_counter()

    # ── собственно fit (в отдельном потоке) ───────────────────────────────
    result: dict[str, object] = {}

    def _run_fit():
        cur_alpha = alpha
        last_exc  = None
        for _ in range(max_ridge_tries):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    mdl = GraphicalLasso(alpha=cur_alpha,
                                         max_iter=max_iter,
                                         tol=tol)
                    mdl.fit(X)
                result["model"] = mdl
                return
            except (FloatingPointError,
                    np.linalg.LinAlgError,
                    ValueError) as e:
                last_exc = e
                cur_alpha *= ridge_factor
        result["exc"] = last_exc

    th = threading.Thread(target=_run_fit, daemon=True)
    th.start()

    # ── progress, пока идёт fit ───────────────────────────────────────────
    while th.is_alive():
        if cancel_cb and cancel_cb():
            raise RuntimeError("cancelled by user")
        if progress:
            frac = min((time.perf_counter() - t0) / t_pred, 0.99)
            progress(offset + span * 0.8 * frac)
        time.sleep(0.2)
    th.join()

    # ── обработка ошибок ─────────────────────────────────────────────────
    if "exc" in result:
        raise RuntimeError(
            f"GraphicalLasso failed: {result['exc']}\n"
            "Try increasing the initial alpha or preprocessing the data "
            "(e.g., stronger scaling / feature filtering)."
        )

    model: GraphicalLasso = result["model"]
    if model.precision_ is None:
        raise RuntimeError("glasso failed to converge")

    # 1) partial corr
    prec  = model.precision_
    d_inv = 1.0 / np.sqrt(np.diag(prec))
    P     = -prec * d_inv[:, None] * d_inv[None, :]
    np.fill_diagonal(P, 0)

    # 2) adjacency + контроль плотности
    adj = np.abs(P) >= thr
    np.fill_diagonal(adj, 0)
    edge_cnt = int(np.count_nonzero(np.triu(adj, k=1)))
    if edge_cnt > MAX_EDGES:
        raise RuntimeError(
            f"Network too dense (limit ≈ {MAX_EDGES}). Increase edgeThreshold."
        )

    # 3) graph
    G = nx.from_pandas_adjacency(
        pd.DataFrame(adj.astype(np.uint8), index=ids, columns=ids)
    )
    if weight_flag == "on":
        absP = np.abs(P)
        idx  = {v: i for i, v in enumerate(ids)}
        for u, v in G.edges():
            G[u][v]["weight"] = float(absP[idx[u], idx[v]])

    if progress:
        progress(offset + span)

    return G, P
                   
BUILD = {"spearman": _corr, "clr": _clr, "rf": _rf, "glasso":   _glasso, }


def _fuse(mats: List[np.ndarray], thr, how="mean"):
    """Element‑wise fuse (mean / median / max) after masking by threshold."""
    masked = [np.where(np.abs(m) >= thr, m, np.nan) for m in mats]
    stack = np.stack(masked, axis=0)

    if how == "mean":
        return np.nanmean(stack, 0)
    if how == "median":
        return np.nanmedian(stack, 0)
    if how == "max":
        return np.nanmax(stack, 0)
    raise ValueError("Unknown fuse mode.")


def _add_cross(Gm, A: pd.DataFrame, B: pd.DataFrame, method, thr):
    """Add cross‑omics edges between two matrices A and B."""
    shared = list(set(A.columns) & set(B.columns))
    if not shared:
        return

    ia, ib = A.index.to_list(), B.index.to_list()
    expr = pd.concat([A.loc[ia, shared], B.loc[ib, shared]], axis=0)
    S = BUILD[method](expr.T[ia + ib], 0, "off")[1]
    block = S[: len(ia), len(ia) :]

    for i, u in enumerate(ia):
        for j, v in enumerate(ib):
            w = float(abs(block[i, j]))
            if w >= thr:
                Gm.add_edge(u, v, layer=f"cross_{method}", weight=w)


# ═════════════════════════════════════════════════════════════════════════════
# 4  API View
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# 3  Worker  — heavy job executed in a background thread
# ═════════════════════════════════════════════════════════════════════════════
def _network_worker(task_id: str,
                    files_bin: list[tuple[bytes, str]],
                    types: list[str],
                    params: dict):
    """
    Full «build‑network» pipeline (identical to your original logic) with
    incremental set_progress(%) calls so the front‑end LinearProgress shows
    real advancement for long CLR / RF computations.
    """
    
    def cancelled():
        with TASKS_LOCK:
            return TASKS.get(task_id, {}).get("cancelled", False)
            
    # ── helper to stream % into TASKS dict ──────────────────────────────────
    def set_progress(val: float):
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id]["progress"] = round(min(max(val, 0), 100), 1)

    try:
        # 0 %‑2 % — thread started
        set_progress(2)

        # 1) ───────── reconstruct uploaded files ───────────────────────────
        files = []
        for raw, name in files_bin:
            buf = io.BytesIO(raw)
            buf.name = name
            files.append(buf)

        # 2) ───────── extract parameters (exactly as original) ─────────────
        sync_all = str(params.get("syncAll", True)).lower() == "true"
        p_data   = params.get("paramData", {})
        net_p    = p_data.get("networkParams", {})
        pre_f    = p_data.get("preFilterParams", {})

        # 3) ───────── load Rodin objects ───────────────────────────────────
        rodins = _load_rodins(files, types)
        set_progress(10)

        if (len(rodins) == 1
                and net_p.get("layerMode", "stack").lower() == "multilayer"):
            raise ValueError(SINGLE_OMIC_GUARD)

        # 4) ───────── preprocessing / filtering & fstats ───────────────────
        fstats = [{"features_original": len(r.features)} for r in rodins]

        def _apply(i, r, pre, filt):
            if pre:
                before = len(r.features)
                r = _preproc(r, pre)
                fstats[i]["features_after_preproc"]   = len(r.features)
                fstats[i]["features_removed_preproc"] = before - len(r.features)
            else:
                fstats[i].update(features_after_preproc=len(r.features),
                                 features_removed_preproc=0)

            if filt:
                before = len(r.features)
                r = _filter(r, filt)
                fstats[i]["features_before_filter"]   = before
                fstats[i]["features_after_filter"]    = len(r.features)
                fstats[i]["features_removed_filter"]  = before - len(r.features)
            else:
                fstats[i].update(features_before_filter=len(r.features),
                                 features_after_filter=len(r.features),
                                 features_removed_filter=0)
            return r

        if sync_all:
            d    = (pre_f[0] if isinstance(pre_f, list) and pre_f else pre_f).get("data", {})
            pre  = {k: d[k] for k in ("threshold","normalization",
                                      "logTransformation","scaling") if k in d}
            filt = {k: d[k] for k in ("method","meta","Threshold") if k in d}
            rodins = [_apply(i, r, pre, filt) for i, r in enumerate(rodins)]
        else:
            tmp = []
            for i, r in enumerate(rodins):
                d    = (pre_f[i] if i < len(pre_f) else {}).get("data", {})
                pre  = {k: d[k] for k in ("threshold","normalization",
                                          "logTransformation","scaling") if k in d}
                filt = {k: d[k] for k in ("method","meta","Threshold") if k in d}
                tmp.append(_apply(i, r, pre, filt))
            rodins = tmp

        for i, r in enumerate(rodins):
            fstats[i]["X_shape"] = list(r.X.shape)

        if sum(len(r.features) for r in rodins) == 0:
            raise ValueError("All features were filtered out.")

        set_progress(20)

        # 5) ───────── network parameters ───────────────────────────────────
        mth     = net_p.get("networkMethod", "spearman").lower()
        lmode   = net_p.get("layerMode",     "stack").lower()
        combine = net_p.get("combineSamples","mean").lower()
        thr     = float(net_p.get("edgeThreshold", 0.75))
        nmode   = net_p.get("nodeMode", "samples").lower()
        wflag   = net_p.get("weights", "on").lower()
        layout  = net_p.get("layout", "force-directed")

        if nmode not in ("samples", "features"):
            raise ValueError("nodeMode must be ‘samples’ or ‘features’.")

        # 6) ───────── feature IDs disambiguation (nodeMode=features) ───────
        if nmode == "features":
            for r in rodins:
                tag = r.uns["file_name"].replace(".", "_")
                r.X.index = r.features.index = [f"{tag}__{fid}" for fid in r.X.index]

        combined = pd.concat([r.X for r in rodins], axis=0)

        def _build(df, offset):
            if mth == "clr":
                # берем n_neighbors из UI
                n_n = int(net_p.get("n_neighbors", 2))
                return _clr(
                    df, thr, wflag,
                    progress=set_progress, offset=offset, span=60,
                    cancel_cb=cancelled,
                    n_neighbors=n_n
                )

            if mth == "rf":
                # параметры из UI: n_estimators и max_depth
                n_est = int(net_p.get("n_estimators", 80))
                m_dep = net_p.get("max_depth", None)
                # если max_depth пустой или "0", превращаем в None
                m_dep = None if m_dep in (None, "", "0") else int(m_dep)
                return _rf(
                    df, thr, wflag,
                    progress=set_progress, offset=offset, span=60,
                    cancel_cb=cancelled,
                    n_estimators=n_est,
                    max_depth=m_dep
                )

            if mth == "glasso":
                # параметры из UI: glassoAlpha и glassoMaxIter
                alpha    = float(net_p.get("glassoAlpha", 0.05))
                max_iter = int(net_p.get("glassoMaxIter", 200))
                return _glasso(
                    df, thr, wflag,
                    progress=set_progress, offset=offset, span=60,
                    cancel_cb=cancelled,
                    alpha=alpha,
                    max_iter=max_iter
                )

            # default — Spearman
            return _corr(df, thr, wflag)

        # 7) ───────── build graph (identical branches) ─────────────────────
        set_progress(30)  # entering builder section

        if nmode == "samples":
            # — Nodes = samples —
            if lmode == "stack":
                G, _ = _build(combined, 30)
                for u, v in G.edges():
                    G[u][v].update(layer="Entire", layers={"Entire"})
            else:  # multilayer
                mats, per_layer_edges = [], defaultdict(set)
                for r in rodins:
                    Gl, W = _build(r.X, 30)
                    mats.append(W)
                    per_layer_edges[r.uns["file_name"]].update(
                        frozenset((u, v)) for u, v in Gl.edges()
                    )

                fused = _fuse(mats, thr, combine)
                ids   = rodins[0].X.columns.to_list()
                adj   = (np.abs(fused) >= thr).astype(int)
                np.fill_diagonal(adj, 0)
                G = nx.from_pandas_adjacency(pd.DataFrame(adj, index=ids, columns=ids))

                for i, u in enumerate(ids):
                    for j in range(i + 1, len(ids)):
                        v = ids[j]
                        if not G.has_edge(u, v):
                            continue
                        if wflag == "on":
                            G[u][v]["weight"] = float(abs(fused[i, j]))

                        layers = {"Entire"}
                        present = [f for f, e in per_layer_edges.items()
                                   if frozenset((u, v)) in e]
                        if len(present) == len(rodins):
                            layers.add("consensus")
                        layers.update(present)
                        G[u][v]["layers"] = set(layers)
                        G[u][v]["layer"]  = ",".join(sorted(layers))
        else:
            # — Nodes = features —
            if lmode == "stack":
                G, _ = _build(combined.T, 30)
                for u, v in G.edges():
                    G[u][v].update(layer="Entire", layers={"Entire"})
            else:
                Gm = nx.MultiGraph()
                for r in rodins:
                    Gl, _ = _build(r.X.T, 30)
                    tag = r.uns["file_name"]
                    for u, v, d in Gl.edges(data=True):
                        Gm.add_edge(u, v, layer=tag, weight=d.get("weight", 1))
                    Gm.add_nodes_from(Gl.nodes(data=True))

                if len(rodins) >= 2:
                    for a, b in itertools.combinations(rodins, 2):
                        _add_cross(Gm, a.X, b.X, mth, thr)

                G = nx.Graph()
                for u, v, d in Gm.edges(data=True):
                    if G.has_edge(u, v):
                        if d["weight"] > G[u][v]["weight"]:
                            G[u][v]["weight"] = d["weight"]
                        G[u][v]["layers"].add(d["layer"])
                    else:
                        G.add_edge(u, v, weight=d["weight"], layers={d["layer"]})
                for u, v in G.edges():
                    G[u][v]["layers"].add("Entire")
                    G[u][v]["layer"] = ",".join(sorted(G[u][v]["layers"]))

        # 8) ───────── self‑loops / density check ───────────────────────────
        G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_edges() > MAX_EDGES:
            raise ValueError(
                f"Network too dense: {G.number_of_edges()} edges "
                f"(limit ≈ {MAX_EDGES}). Increase edgeThreshold."
            )

        # 9) ───────── node metadata (original logic) ───────────────────────
        if nmode == "samples":
            sample_df = next(
                (
                    r.samples
                    for r in rodins
                    if r.samples is not None and not r.samples.empty
                ),
                pd.DataFrame({"id": list(G.nodes())}),
            )
            meta = sample_df.set_index(sample_df.columns[0]).to_dict("index")
            for n in G.nodes():
                G.nodes[n].update(meta.get(n, {}))
        else:
            file_map, type_map, compound_map = {}, {}, {}
            for r in rodins:
                fn, ft = r.uns["file_name"], r.uns["file_type"]
                lead = 2 if ft == "metabolomics" else 1
                for fid in r.X.index:
                    file_map.setdefault(fid, set()).add(fn)
                    type_map.setdefault(fid, set()).add(ft)
                for fid in r.features.index:
                    compound_map[fid] = "_".join(
                        r.features.loc[fid].iloc[:lead].astype(str)
                    )

            for n in G.nodes():
                G.nodes[n]["file"]     = ",".join(sorted(file_map.get(n, [])))
                G.nodes[n]["type"]     = ",".join(sorted(type_map.get(n, [])))
                G.nodes[n]["compound"] = compound_map.get(n, "")

        # 10) ───────── layout + communities ────────────────────────────────
        layouts = {
            "force-directed": nx.spring_layout,
            "spring":         nx.spring_layout,
            "circular":       nx.circular_layout,
            "kamada_kawai":   nx.kamada_kawai_layout,
            "random":         nx.random_layout,
        }
        if layout == "force-directed":
            pos = layouts[layout](G, seed=777)
        else:
            pos = layouts.get(layout, nx.spring_layout)(G)

        for n in G.nodes():
            G.nodes[n]["x"], G.nodes[n]["y"] = map(float, pos[n])
            G.nodes[n]["display_id"] = (
                n.split("__")[-1] if nmode == "features" else n
            )

        comps = list(nx.connected_components(G))
        label = 2
        for comp in comps:
            comp = list(comp)
            if len(comp) == 1:
                G.nodes[comp[0]]["community"] = "Group_I"
            else:
                for v in comp:
                    G.nodes[v]["community"] = f"Group_{label}"
                label += 1

        # 11) ───────── stats (exact copy) ──────────────────────────────────
        deg  = dict(G.degree())
        active_nodes = [n for n, d in deg.items() if d > 0]

        nodes_with_edges = len(active_nodes)
        density_active   = (round(nx.density(G.subgraph(active_nodes)), 4)
                            if len(active_nodes) > 1 else 0)
        density_all      = round(nx.density(G), 4)

        nstats = {
            "numNodes":          G.number_of_nodes(),
            "numEdges":          G.number_of_edges(),
            "nodesWithEdges":    nodes_with_edges,
            "numNodesWithEdges": nodes_with_edges,
            "density_":          density_active,
            "density":           density_all,
            "numComponents":     nx.number_connected_components(G),
            "numCommunities":    nx.number_connected_components(G)
                                 - sum(1 for d in deg.values() if d < 1),
        }

        if lmode == "multilayer":
            l_nodes, l_edges = defaultdict(set), defaultdict(int)
            for u, v, d in G.edges(data=True):
                for lay in d["layers"]:
                    if lay == "Entire":
                        continue
                    l_nodes[lay].update([u, v])
                    l_edges[lay] += 1

            for lay in sorted(l_edges):
                safe = lay.replace(" ", "_").replace(".", "_").replace(",", "_")
                sub  = G.subgraph(l_nodes[lay])
                nstats[f"nodes_{safe}"]   = len(l_nodes[lay])
                nstats[f"edges_{safe}"]   = l_edges[lay]
                nstats[f"density_{safe}"] = (
                    round(nx.density(sub), 4) if len(l_nodes[lay]) > 1 else 0
                )

            # consensus layer
            c_edges = [(u, v) for u, v in G.edges()
                       if "consensus" in G[u][v]["layers"]]
            if c_edges:
                Gc = G.edge_subgraph(c_edges).copy()
                nstats.update(
                    nodes_consensus   = Gc.number_of_nodes(),
                    edges_consensus   = Gc.number_of_edges(),
                    density_consensus = round(nx.density(Gc), 4),
                )

        # 12) ───────── JSON payload ────────────────────────────────────────
        nodes_json = [
            {"id": n, "x": d["x"], "y": d["y"],
             **{k: v for k, v in d.items() if k not in ("x", "y")}}
            for n, d in G.nodes(data=True)
        ]
        edges_json = [
            {"source": u, "target": v,
             "weight": float(G[u][v].get("weight", 1)),
             "layer":  G[u][v]["layer"],
             "layers": sorted(G[u][v]["layers"])}
            for u, v in G.edges()
        ]
        if not edges_json:
            raise ValueError("No edges (edgeThreshold too high?).")


                # ───────── подготовка union-наборов мета- и sample-полей ─────────
        union_meta_cols   = set()
        union_sample_cols = set()
        for r in rodins:
            if r.features is None or r.features.empty:
                continue
            union_meta_cols.update(r.features.columns)
            union_sample_cols.update(r.X.index)

        # Собираем объединённые наборы meta- и sample-полей
        meta_cols   = set(col for r in rodins if r.features is not None for col in r.features.columns)
        sample_cols = set(idx for r in rodins for idx in r.X.index)

        # Формируем список по-файловых таблиц
        dfs = [
            pd.concat([r.features, r.X], axis=1, join='outer')
            .assign(file=r.uns.get('file_name', ''))
            .reset_index().rename(columns={'index': 'feature'})
            for r in rodins
            if r.features is not None and not r.features.empty
        ]

        if dfs:
            df_out = pd.concat(dfs, ignore_index=True)
            # Порядок колонок: feature, file, все meta_cols, все sample_cols, остальные
            all_cols = df_out.columns.tolist()
            ordered = ['feature', 'file'] \
                    + [c for c in all_cols if c in meta_cols] \
                    + [c for c in all_cols if c in sample_cols] \
                    + [c for c in all_cols if c not in meta_cols | sample_cols | {'feature', 'file'}]
            df_out = df_out[ordered]
            # Конвертация numpy → Python, NaN → None
            raw = df_out.to_dict(orient='records')
            data_table = [
                {k: (None if pd.isna(v)
                    else v.item() if isinstance(v, np.generic)
                    else v)
                for k, v in rec.items()}
                for rec in raw
            ]
        else:
            data_table = []

        payload = {
            "networkMethod": mth,
            "edgeThreshold": thr,
            "layout":       layout,
            "layerMode":    lmode,
            "nodes":        nodes_json,
            "edges":        edges_json,
            "stats":        {"fileStats": fstats, "networkStats": nstats},
            "dataTable":    data_table,
        }

        set_progress(100)
        with TASKS_LOCK:
            TASKS[task_id].update(status="done", result=payload)

    # ───────── error handling inside worker ────────────────────────────────
    except Exception as exc:   # noqa: BLE001
        with TASKS_LOCK:
         TASKS[task_id].update(status="error",
                               progress=100,
                               error=str(exc))

# ═════════════════════════════════════════════════════════════════════════════
# 4  API View (unchanged from previous answer — dual POST / GET)
# ═════════════════════════════════════════════════════════════════════════════
class BuildNetworkView(APIView):
    """
    POST (multipart)            → creates background task, returns task_id
    GET  ?task_id=<uuid>        → progress / result / error
    """

    def post(self, request, format=None):
        if request.GET.get("action") == "cancel":
            return self.patch(request, format=format)
        try:
            files = request.FILES.getlist("data_files")
            types = request.data.getlist("file_types")
            if not files or len(files) != len(types):
                raise ValueError("Upload files + file_types mismatch.")

            params = json.loads(request.data.get("parameters", "{}"))

            # copy files → bytes (because Django deletes them after response)
            files_bin = [(f.read(), getattr(f, "name", f"file{i}"))
                         for i, f in enumerate(files)]

            task_id = uuid.uuid4().hex
            with TASKS_LOCK:
                TASKS[task_id] = {"status": "running", "progress": 0,"cancelled": False}

            threading.Thread(
                target=_network_worker,
                args=(task_id, files_bin, types, params),
                daemon=True,
            ).start()

            return JsonResponse({"task_id": task_id}, status=202)

        except ValueError as ve:
            return JsonResponse({"error": str(ve)},
                                status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            return JsonResponse({"error": str(exc)},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request, format=None):
        task_id = request.GET.get("task_id")
        if not task_id:
            return JsonResponse({"error": "Missing task_id parameter."},
                                status=status.HTTP_400_BAD_REQUEST)

        with TASKS_LOCK:
            info = TASKS.get(task_id)

        if info is None:
            return JsonResponse({"error": "task_id not found."},
                                status=status.HTTP_404_NOT_FOUND)

        resp = {"status": info["status"], "progress": info["progress"]}
        if info["status"] == "done":
            resp["result"] = info["result"]
        elif info["status"] == "error":
            resp["error"] = info["error"]
        return JsonResponse(resp, status=200)
        
    def patch(self, request, format=None):
        """
        PATCH ?action=cancel&task_id=<uuid>  → пометить задачу как отменённую
        """
        task_id = request.GET.get("task_id")
        if request.GET.get("action") != "cancel" or not task_id:
            return JsonResponse(
                {"error": "Need task_id & action=cancel"},
                status=status.HTTP_400_BAD_REQUEST
            )

        with TASKS_LOCK:
            info = TASKS.get(task_id)
            if info is None:
                return JsonResponse(
                    {"error": "task_id not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            info["cancelled"] = True
            if info["status"] == "running":
                info["status"] = "cancelling"

        return JsonResponse({"status": "cancelling"}, status=202)

