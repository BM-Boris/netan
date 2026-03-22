"""Internal build and inference helpers for Netan."""

from __future__ import annotations

import copy
import re
import threading
import time
import warnings
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
from networkx.algorithms.community import louvain_communities
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import GraphicalLasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm

MAX_EDGES = 10_000
_CORE_GRAPHS = ("entire", "fused", "consensus", "cross")


def _build_state(
    *,
    graphs: Dict[str, nx.Graph],
    infos: Dict[str, Dict[str, Any]],
    matrices: Dict[str, Dict[str, np.ndarray]],
    tag_map: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "graphs": dict(graphs),
        "infos": {str(k): dict(v) for k, v in infos.items()},
        "matrices": dict(matrices),
        "tag_map": dict(tag_map),
    }


def _network_stats(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], dtype=float)
    active = deg > 0
    n_active = int(active.sum())
    density_all = float((2 * m) / (n * (n - 1))) if n > 1 else 0.0
    density_active = float((2 * m) / (n_active * (n_active - 1))) if n_active > 1 else 0.0
    mean_deg = float(deg.mean()) if n else 0.0
    median_deg = float(np.median(deg)) if n else 0.0
    max_deg = float(deg.max()) if n else 0.0
    mean_deg_active = float(deg[active].mean()) if n_active else 0.0
    median_deg_active = float(np.median(deg[active])) if n_active else 0.0
    max_deg_active = float(deg[active].max()) if n_active else 0.0
    comms = {d.get("community", "Community_0") for _, d in G.nodes(data=True)}
    mods = {d.get("module", "Module_0") for _, d in G.nodes(data=True)}
    return {
        "nodes": n,
        "edges": m,
        "numNodes": n,
        "numEdges": m,
        "nodesWithEdges": n_active,
        "density": density_all,
        "densityAll": density_all,
        "densityActive": density_active,
        "communities": len(comms),
        "modules": len(mods),
        "numCommunities": len(comms),
        "numModules": len(mods),
        "meanDegree": mean_deg,
        "medianDegree": median_deg,
        "maxDegree": max_deg,
        "meanDegreeActive": mean_deg_active,
        "medianDegreeActive": median_deg_active,
        "maxDegreeActive": max_deg_active,
    }

def _ensure_df(x, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas.DataFrame")
    return x

def _finalize_matrix(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Expected a square similarity matrix.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr + arr.T) * 0.5
    np.fill_diagonal(arr, 0.0)
    return arr

def _corr_matrix(df: pd.DataFrame) -> np.ndarray:
    cor = df.corr("spearman").fillna(0.0)
    return _finalize_matrix(cor.to_numpy(dtype=np.float32, copy=False))

def _clr_matrix(
    df: pd.DataFrame,
    *,
    n_jobs: int,
    n_neighbors: int = 2,
) -> np.ndarray:
    """Fast CLR (symmetric MI z-scores) without graph construction."""
    X = df.values.astype("float32", copy=False)
    p = X.shape[1]
    if p <= 1:
        return np.zeros((p, p), dtype=np.float32)

    MI = np.zeros((p, p), dtype=np.float32)

    def mi_column(j: int):
        return mutual_info_regression(
            X,
            X[:, j],
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=0,
        )

    chunk = max(1, min(8, p))
    with tqdm(total=p, desc="CLR", leave=False) as bar:
        for start in range(0, p, chunk):
            cols = range(start, min(start + chunk, p))
            res = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(mi_column)(j) for j in cols
            )
            MI[:, cols] = np.column_stack(res)
            bar.update(len(list(cols)))

    MI = _finalize_matrix(MI)
    mu = MI.mean(1, keepdims=True)
    sig = MI.std(1, keepdims=True) + 1e-9
    z = (MI - mu) / sig
    S = np.sqrt(z ** 2 + z.T ** 2)
    return _finalize_matrix(S)

def _rf_matrix(
    df: pd.DataFrame,
    *,
    n_jobs: int,
    n_estimators: int = 160,
    max_depth: Optional[int] = None,
) -> np.ndarray:
    """ExtraTrees-based symmetric importance matrix without graph construction."""
    X = df.values.astype("float32", copy=False)
    p = X.shape[1]
    if p <= 1:
        return np.zeros((p, p), dtype=np.float32)

    W = np.zeros((p, p), dtype=np.float32)
    if max_depth in (0, "0", ""):
        max_depth = None

    def fit_target(t: int):
        y = X[:, t]
        Xo = np.delete(X, t, axis=1)
        mdl = ExtraTreesRegressor(
            n_estimators=int(n_estimators),
            max_depth=(None if max_depth is None else int(max_depth)),
            random_state=1,
            max_features="sqrt",
        )
        mdl.fit(Xo, y)
        row = np.zeros(p, dtype=np.float32)
        row[np.arange(p) != t] = mdl.feature_importances_
        return t, row

    chunk = max(1, min(4, p))
    with tqdm(total=p, desc="RF", leave=False) as bar:
        for start in range(0, p, chunk):
            idxs = range(start, min(start + chunk, p))
            rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(fit_target)(t) for t in idxs
            )
            for t, row in rows:
                W[t] = row
            bar.update(len(list(idxs)))

    return _finalize_matrix(W)

def _glasso_matrix(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-4,
    ridge_factor: float = 10.0,
    max_ridge_tries: int = 8,
) -> np.ndarray:
    """Graphical Lasso partial-correlation matrix without graph construction."""
    X = df.values.astype("float32", copy=False)
    p = X.shape[1]
    if p <= 1:
        return np.zeros((p, p), dtype=np.float32)

    CALIB_P = 30
    MIN_T_PRED = 0.5

    def _estimate_runtime() -> float:
        p_sub = min(p, CALIB_P)
        if p_sub <= 1:
            return MIN_T_PRED
        Xsub = X[:, :p_sub]
        t0 = time.perf_counter()
        mdl = GraphicalLasso(alpha=alpha, max_iter=max(1, max_iter // 4), tol=1e-3).fit(Xsub)
        dt = max(time.perf_counter() - t0, 1e-3)
        k = dt / (p_sub ** 3 * max(1, max_iter // 4))
        return max(k * p ** 3 * max_iter, MIN_T_PRED)

    t_pred = _estimate_runtime()
    result: Dict[str, object] = {}

    def _run_fit():
        cur_alpha = float(alpha)
        last_exc = None
        for _ in range(int(max_ridge_tries)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    mdl = GraphicalLasso(alpha=cur_alpha, max_iter=int(max_iter), tol=float(tol))
                    mdl.fit(X)
                result["model"] = mdl
                return
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as e:
                last_exc = e
                cur_alpha *= float(ridge_factor)
        result["exc"] = last_exc

    th = threading.Thread(target=_run_fit, daemon=True)
    th.start()

    with tqdm(total=100, desc="Glasso", leave=False) as bar:
        t0 = time.perf_counter()
        while th.is_alive():
            frac = min((time.perf_counter() - t0) / max(1e-6, t_pred), 0.99)
            bar.n = int(frac * 100)
            bar.refresh()
            time.sleep(0.2)
        th.join()
        bar.n = 100
        bar.refresh()

    if "exc" in result:
        raise RuntimeError(
            f"GraphicalLasso failed: {result['exc']}\n"
            "Try increasing alpha or preprocessing data."
        )

    model: GraphicalLasso = result["model"]  # type: ignore[assignment]
    if model.precision_ is None:
        raise RuntimeError("glasso failed to converge")

    prec = model.precision_
    d_inv = 1.0 / np.sqrt(np.diag(prec))
    P = -prec * d_inv[:, None] * d_inv[None, :]
    return _finalize_matrix(P)

BUILDERS = {
    "spearman": _corr_matrix,
    "clr": _clr_matrix,
    "rf": _rf_matrix,
    "glasso": _glasso_matrix,
}

def _normalize_matrix(W_raw: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    W_abs = np.abs(_finalize_matrix(W_raw))
    n = W_abs.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    if n <= 1:
        return out

    tri = np.triu_indices(n, k=1)
    if mask is None:
        tri_mask = np.ones(tri[0].shape[0], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != W_abs.shape:
            raise ValueError("mask must match the shape of W_raw.")
        tri_mask = mask[tri]
    vals = W_abs[tri][tri_mask]
    if vals.size == 0 or np.all(vals <= 0):
        return out

    if vals.size == 1:
        scaled = np.array([1.0 if vals[0] > 0 else 0.0], dtype=np.float32)
    elif np.allclose(vals, vals[0]):
        scaled = np.ones_like(vals, dtype=np.float32) if vals[0] > 0 else np.zeros_like(vals, dtype=np.float32)
    else:
        ranks = pd.Series(vals).rank(method="average").to_numpy(dtype=np.float32) - 1.0
        scaled = ranks / max(vals.size - 1, 1)

    tri_i = tri[0][tri_mask]
    tri_j = tri[1][tri_mask]
    out[(tri_i, tri_j)] = scaled
    out[(tri_j, tri_i)] = scaled
    np.fill_diagonal(out, 0.0)
    return out

def _combine_matrix_stack(mats: List[np.ndarray], how: str = "mean") -> np.ndarray:
    if not mats:
        raise ValueError("At least one matrix is required for fusion.")
    stack = np.stack([_finalize_matrix(m) for m in mats], axis=0)
    if how == "mean":
        out = np.mean(stack, axis=0)
    elif how == "median":
        out = np.median(stack, axis=0)
    elif how == "max":
        out = np.max(stack, axis=0)
    else:
        raise ValueError("Unknown fuse mode.")
    return _finalize_matrix(out)

def _adjacency_stats(adj: np.ndarray) -> Dict[str, float]:
    adj = np.asarray(adj, dtype=bool)
    n = int(adj.shape[0])
    edges = int(np.count_nonzero(np.triu(adj, k=1)))
    degree = np.asarray(adj.sum(axis=1), dtype=int)
    active_nodes = int(np.count_nonzero(degree))
    active_degree = degree[degree > 0]
    isolate_count = int(n - active_nodes)
    density_all = (
        float((2 * edges) / (n * (n - 1)))
        if n > 1
        else 0.0
    )
    density_active = (
        float((2 * edges) / (active_nodes * (active_nodes - 1)))
        if active_nodes > 1
        else 0.0
    )
    if active_nodes:
        H = nx.from_numpy_array(adj)
        largest_component = max((len(comp) for comp in nx.connected_components(H)), default=0)
        largest_component_fraction = float(largest_component / active_nodes)
    else:
        largest_component_fraction = 0.0
    return {
        "nodes": n,
        "edges": edges,
        "active_nodes": active_nodes,
        "active_fraction": float(active_nodes / n) if n else 0.0,
        "isolate_count": isolate_count,
        "density_all": density_all,
        "density_active": density_active,
        "largest_component_fraction": largest_component_fraction,
        "mean_degree": float((2 * edges) / n) if n else 0.0,
        "mean_degree_active": float(active_degree.mean()) if active_degree.size else 0.0,
        "median_degree_active": float(np.median(active_degree)) if active_degree.size else 0.0,
    }

def _knn_adjacency(allowed: np.ndarray, W_norm: np.ndarray, k: int, mutual: bool) -> np.ndarray:
    knn_mask = np.zeros_like(allowed, dtype=bool)
    for i in range(allowed.shape[0]):
        idx = np.flatnonzero(allowed[i])
        if idx.size == 0:
            continue
        if idx.size > k:
            order = np.argsort(-W_norm[i, idx], kind="mergesort")[:k]
            idx = idx[order]
        knn_mask[i, idx] = True
    adj = allowed & (knn_mask & knn_mask.T if mutual else (knn_mask | knn_mask.T))
    np.fill_diagonal(adj, False)
    return adj

def _auto_k(n: int, stats: Dict[str, float]) -> Optional[int]:
    active_nodes = int(stats.get("active_nodes", 0))
    if n <= 1 or active_nodes <= 2:
        return None

    mean_active = float(stats.get("mean_degree_active", 0.0))
    median_active = float(stats.get("median_degree_active", 0.0))

    if mean_active <= 5.0:
        return None

    if active_nodes <= 12 and mean_active <= 7.0:
        return None

    k = 3 + int(np.floor(np.sqrt(max(mean_active - 5.0, 0.0))))
    k_cap = max(3, int(np.ceil(median_active)))

    return min(max(3, k), k_cap, active_nodes - 1, 10)

def _choose_threshold_from_norm(W_norm: np.ndarray, auto_target: float = 0.95) -> Dict[str, float]:
    W_norm = _finalize_matrix(W_norm)
    n = int(W_norm.shape[0])
    target_fraction = float(auto_target)
    target_nodes = int(np.ceil(target_fraction * n)) if n else 0
    strategy = f"target{100.0 * target_fraction:g}"
    if n <= 1:
        return {
            "thr_norm": 1.0,
            "strategy": strategy,
            "edges": 0,
            "active_nodes": 0,
            "active_fraction": 0.0,
            "largest_component_fraction": 0.0,
            "target_nodes": target_nodes,
        }

    def _mask(thr: float) -> np.ndarray:
        allowed = W_norm >= thr
        np.fill_diagonal(allowed, False)
        return allowed

    tri_vals = W_norm[np.triu_indices(n, k=1)]
    positive = np.unique(tri_vals[tri_vals > 0])
    if positive.size == 0:
        stats = _adjacency_stats(np.zeros_like(W_norm, dtype=bool))
        thr = 1.0
    else:
        lo, hi = 0, positive.size - 1
        best_idx = None
        best_stats = None
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = float(positive[mid])
            cand_stats = _adjacency_stats(_mask(cand))
            if cand_stats["active_nodes"] >= target_nodes:
                best_idx = mid
                best_stats = cand_stats
                lo = mid + 1
            else:
                hi = mid - 1
        if best_idx is None:
            thr = float(positive[0])
            stats = _adjacency_stats(_mask(thr))
        else:
            thr = float(positive[best_idx])
            stats = best_stats

    return {
        "thr_norm": float(thr),
        "strategy": strategy,
        "edges": stats["edges"],
        "active_nodes": stats["active_nodes"],
        "active_fraction": stats["active_fraction"],
        "largest_component_fraction": stats["largest_component_fraction"],
        "target_nodes": target_nodes,
    }

def _reattach_isolates(adj: np.ndarray, W_norm: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    adj = np.asarray(adj, dtype=bool).copy()
    isolate_before = np.flatnonzero(~adj.any(axis=1))
    attached = 0
    for i in isolate_before.tolist():
        scores = np.asarray(W_norm[i], dtype=np.float32).copy()
        scores[i] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]) or scores[j] <= 0:
            continue
        adj[i, j] = True
        adj[j, i] = True
        attached += 1
    np.fill_diagonal(adj, False)
    isolate_after = np.flatnonzero(~adj.any(axis=1))
    return adj, {
        "isolate_count_before": int(isolate_before.size),
        "isolate_count": int(isolate_after.size),
        "isolates_reattached": int(attached),
        "isolates_unreattached": int(isolate_after.size),
        "used_isolate_reattachment": bool(isolate_before.size),
    }

def _effective_threshold_pair(
    W_abs: np.ndarray,
    W_norm: np.ndarray,
    allowed: np.ndarray,
    *,
    thr_raw_input: Optional[float],
    thr_norm_input: Optional[float],
) -> Tuple[float, float]:
    tri_mask = np.triu(np.ones_like(allowed, dtype=bool), k=1)
    pos_mask = tri_mask & (W_abs > 0)
    allowed_mask = pos_mask & np.asarray(allowed, dtype=bool)
    if np.any(allowed_mask):
        return float(W_abs[allowed_mask].min()), float(W_norm[allowed_mask].min())

    pos_abs = W_abs[pos_mask]
    pos_norm = W_norm[pos_mask]
    raw_eff = float(thr_raw_input) if thr_raw_input is not None else (float(pos_abs.max()) if pos_abs.size else 0.0)
    norm_eff = float(thr_norm_input) if thr_norm_input is not None else (float(pos_norm.max()) if pos_norm.size else 1.0)
    return raw_eff, norm_eff

def _sparsify_matrix(
    W_raw: np.ndarray,
    W_norm: np.ndarray,
    *,
    thr_raw: Optional[float] = None,
    thr_norm: Optional[float] = None,
    auto_target: float = 0.95,
    attach_isolates: bool = False,
    k: Optional[Union[int, str]] = "auto",
    mutual: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    W_abs = np.abs(_finalize_matrix(W_raw))
    W_norm = _finalize_matrix(W_norm)
    if W_abs.shape != W_norm.shape:
        raise ValueError("W_raw and W_norm must have matching shapes.")

    if thr_raw is not None:
        if isinstance(thr_raw, bool) or not isinstance(thr_raw, Real):
            raise TypeError("thr_raw must be a number or None.")
        if float(thr_raw) < 0:
            raise ValueError("thr_raw must be >= 0.")
    if thr_norm is not None:
        if isinstance(thr_norm, bool) or not isinstance(thr_norm, Real):
            raise TypeError("thr_norm must be a number or None.")
        if not 0 <= float(thr_norm) <= 1:
            raise ValueError("thr_norm must be within [0, 1].")
    if isinstance(auto_target, bool) or not isinstance(auto_target, Real):
        raise TypeError("auto_target must be a number within (0, 1].")
    if not 0 < float(auto_target) <= 1:
        raise ValueError("auto_target must be within (0, 1].")
    if not isinstance(attach_isolates, bool):
        raise TypeError("attach_isolates must be a bool.")
    if k is not None:
        if isinstance(k, str):
            if k != "auto":
                raise TypeError("k must be an integer, None, or 'auto'.")
        elif isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k must be an integer, None, or 'auto'.")
        elif int(k) <= 0:
            raise ValueError("k must be >= 1 when provided.")

    auto_info = None
    thr_raw_input = None if thr_raw is None else float(thr_raw)
    thr_norm_input = None if thr_norm is None else float(thr_norm)
    thr_raw_eff = thr_raw_input
    thr_norm_eff = thr_norm_input
    if thr_raw_eff is None and thr_norm_eff is None:
        auto_info = _choose_threshold_from_norm(W_norm, auto_target=float(auto_target))
        thr_norm_eff = float(auto_info["thr_norm"])

    allowed = W_abs > 0
    if thr_raw_eff is not None:
        allowed &= W_abs >= thr_raw_eff
    if thr_norm_eff is not None:
        allowed &= W_norm >= thr_norm_eff
    np.fill_diagonal(allowed, False)
    thr_raw_show, thr_norm_show = _effective_threshold_pair(
        W_abs,
        W_norm,
        allowed,
        thr_raw_input=thr_raw_input,
        thr_norm_input=thr_norm_eff,
    )
    threshold_stats = _adjacency_stats(allowed)

    if k == "auto":
        k_eff = _auto_k(W_abs.shape[0], threshold_stats)
    else:
        k_eff = None if k is None else int(k)
    mutual_eff = bool(mutual) if k_eff is not None else False

    adj = allowed if k_eff is None else _knn_adjacency(allowed, W_norm, k_eff, mutual_eff)

    np.fill_diagonal(adj, False)
    if attach_isolates:
        adj, isolate_info = _reattach_isolates(adj, W_norm)
    else:
        isolate_count = int(np.count_nonzero(~adj.any(axis=1)))
        isolate_info = {
            "isolate_count_before": isolate_count,
            "isolate_count": isolate_count,
            "isolates_reattached": 0,
            "isolates_unreattached": isolate_count,
            "used_isolate_reattachment": False,
        }
    info = {
        "thr_raw": thr_raw_show,
        "thr_norm": thr_norm_show,
        "thr_raw_input": thr_raw_input,
        "thr_norm_input": thr_norm_input,
        "auto_info": auto_info,
        "auto_target": float(auto_target),
        "attach_isolates": bool(attach_isolates),
        "k": k_eff,
        "k_mode": ("auto" if k == "auto" else ("none" if k_eff is None else "explicit")),
        "mutual": mutual_eff,
    }
    info.update(isolate_info)
    return adj, info

def _matrix_to_graph(ids: Sequence[str], W_raw: np.ndarray, adj: np.ndarray) -> nx.Graph:
    nodes = list(map(str, ids))
    W_abs = np.abs(_finalize_matrix(W_raw))
    adj = np.asarray(adj, dtype=bool)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    tri_i, tri_j = np.where(np.triu(adj, k=1))
    G.add_weighted_edges_from(
        (nodes[i], nodes[j], float(W_abs[i, j]))
        for i, j in zip(tri_i.tolist(), tri_j.tolist())
    )
    return G

def _set_single_layer_attrs(G: nx.Graph, layer_name: str):
    for _, _, d in G.edges(data=True):
        d["layer"] = str(layer_name)
        d["layers"] = {str(layer_name)}
        d["support_count"] = 1

def _annotate_support_layers(
    G: nx.Graph,
    ids: Sequence[str],
    layer_adj: Dict[str, np.ndarray],
    *,
    graph_label: str,
):
    idx = {str(node): i for i, node in enumerate(ids)}
    for u, v, d in G.edges(data=True):
        i = idx[str(u)]
        j = idx[str(v)]
        support = [name for name, adj in layer_adj.items() if bool(adj[i, j])]
        layers = {str(graph_label), *map(str, support)}
        d["layers"] = layers
        d["layer"] = ",".join(sorted(layers))
        d["support_count"] = len(support)

def _annotate_feature_layers(
    G: nx.Graph,
    node_to_layer: Dict[str, str],
    *,
    graph_label: str,
):
    for u, v, d in G.edges(data=True):
        lu = node_to_layer.get(str(u))
        lv = node_to_layer.get(str(v))
        layers = {str(graph_label)}
        if lu and lv and lu != lv:
            layers.add("cross")
            d["layer"] = "cross"
            d["support_count"] = 2
        else:
            primary = str(lu or lv or graph_label)
            layers.add(primary)
            d["layer"] = primary
            d["support_count"] = 1
        d["layers"] = layers

def _union_graphs_with_label(
    graphs: Sequence[nx.Graph],
    *,
    graph_label: str,
) -> nx.Graph:
    H = nx.Graph()
    for G in graphs:
        H.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            attrs = dict(d)
            layers = set(attrs.get("layers") or {attrs.get("layer", graph_label)})
            layers.add(str(graph_label))
            attrs["layers"] = layers
            if not attrs.get("layer"):
                attrs["layer"] = str(graph_label)
            H.add_edge(u, v, **attrs)
    return H

def _matrix_cache_entry(
    W_raw: np.ndarray,
    W_norm: np.ndarray,
    *,
    W_abs: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    W_raw = _finalize_matrix(W_raw)
    return {
        "W_raw": W_raw,
        "W_abs": np.abs(W_raw) if W_abs is None else _finalize_matrix(W_abs),
        "W_norm": _finalize_matrix(W_norm),
    }

def _support_base_info(
    infos: Sequence[Dict[str, Any]],
    *,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    infos = [dict(info) for info in infos if info]
    base = dict(fallback or (infos[0] if infos else {}))
    if not infos:
        return base

    def common(key: str, default=None):
        vals = [info.get(key) for info in infos]
        first = vals[0]
        return first if all(val == first for val in vals) else default

    for key in ("thr_raw_input", "thr_norm_input", "auto_target", "attach_isolates", "mutual", "combine", "min_layers"):
        val = common(key, base.get(key))
        if val is not None or key in base:
            base[key] = val

    base["auto_info"] = common("auto_info", None)
    k_modes = [info.get("k_mode") for info in infos]
    base["k_mode"] = k_modes[0] if k_modes and all(mode == k_modes[0] for mode in k_modes) else None
    k_vals = [info.get("k") for info in infos]
    base["k"] = k_vals[0] if k_vals and all(val == k_vals[0] for val in k_vals) else None
    return base

def _infer_graph_bundle(
    df: pd.DataFrame,
    ids: Sequence[str],
    *,
    method: str,
    n_jobs: int,
    infer_kwargs: Dict[str, Any],
    sparsify_kwargs: Dict[str, Any],
) -> Tuple[nx.Graph, Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    W_raw = _dispatch_infer(method, df, n_jobs, **infer_kwargs)
    W_norm = _normalize_matrix(W_raw)
    adj, info = _sparsify_matrix(W_raw, W_norm, **sparsify_kwargs)
    mats = _matrix_cache_entry(W_raw, W_norm)
    return _matrix_to_graph(ids, mats["W_raw"], adj), mats, adj, info

def _build_consensus_graph(
    ids: Sequence[str],
    raw_by_layer: Dict[str, np.ndarray],
    adj_by_layer: Dict[str, np.ndarray],
    *,
    min_layers: int,
    combine: str,
) -> Tuple[nx.Graph, np.ndarray]:
    support_counts = np.zeros_like(next(iter(adj_by_layer.values())), dtype=np.int16)
    for adj in adj_by_layer.values():
        support_counts += adj.astype(np.int16)
    con_adj = support_counts >= int(min_layers)
    np.fill_diagonal(con_adj, False)

    G = nx.Graph()
    G.add_nodes_from(ids)
    tri_i, tri_j = np.where(np.triu(con_adj, k=1))
    G.add_weighted_edges_from(
        (
            ids[i],
            ids[j],
            float(np.max(vals) if combine == "max" else np.median(vals) if combine == "median" else np.mean(vals)),
        )
        for i, j in zip(tri_i.tolist(), tri_j.tolist())
        for vals in [[float(raw_by_layer[name][i, j]) for name, adj in adj_by_layer.items() if bool(adj[i, j])]]
    )
    return G, con_adj

def _tag_feature_frames(
    names: Sequence[str],
    rodins: Sequence[object],
) -> Tuple[List[pd.DataFrame], Dict[str, str], Dict[str, str]]:
    tagged = []
    tag_map: Dict[str, str] = {}
    node_to_layer: Dict[str, str] = {}
    for nm, r in zip(names, rodins):
        tag = (str(nm) or "layer").replace(".", "_")
        tag_map[tag] = str(nm)
        X = r.X.copy()
        X.index = [f"{tag}__{fid}" for fid in X.index.astype(str)]
        tagged.append(X)
        node_to_layer.update({str(node_id): str(nm) for node_id in X.index.astype(str)})
    return tagged, tag_map, node_to_layer

def _infer_feature_cross_matrix(
    tagged: Sequence[pd.DataFrame],
    *,
    method: str,
    n_jobs: int,
    infer_kwargs: Dict[str, Any],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if not tagged:
        return [], np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    ids = [str(fid) for Xtag in tagged for fid in Xtag.index.astype(str)]
    starts = []
    offset = 0
    for Xtag in tagged:
        width = int(Xtag.shape[0])
        starts.append((offset, offset + width))
        offset += width

    W_cross_raw = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for i in range(len(tagged)):
        for j in range(i + 1, len(tagged)):
            df_pair = pd.concat([tagged[i], tagged[j]], axis=0).T
            W_pair = _dispatch_infer(method, df_pair, n_jobs, **infer_kwargs)
            a0, a1 = starts[i]
            b0, b1 = starts[j]
            wa = slice(0, a1 - a0)
            wb = slice(a1 - a0, (a1 - a0) + (b1 - b0))
            block = W_pair[wa, wb]
            W_cross_raw[a0:a1, b0:b1] = block
            W_cross_raw[b0:b1, a0:a1] = block.T

    cross_mask = np.zeros_like(W_cross_raw, dtype=bool)
    for i, (a0, a1) in enumerate(starts):
        for j, (b0, b1) in enumerate(starts):
            if i == j:
                continue
            cross_mask[a0:a1, b0:b1] = True

    W_cross_raw = _finalize_matrix(W_cross_raw)
    W_cross_norm = _normalize_matrix(W_cross_raw, mask=cross_mask)
    return ids, W_cross_raw, W_cross_norm

def _graph_edge_table(G: nx.Graph, node_mode: str) -> pd.DataFrame:
    nodes = dict(G.nodes(data=True))
    rows = []
    for u, v, d in G.edges(data=True):
        lays = d.get("layers", None)
        if lays is None:
            lays = {d.get("layer", "entire")}
        elif not isinstance(lays, (set, list, tuple)):
            lays = {lays}
        layers_list = sorted(map(str, list(lays)))
        row = {
            "source": str(u),
            "target": str(v),
            "weight": float(d.get("weight", 1.0)),
            "layer": d.get("layer") or ",".join(layers_list),
            "layers": "|".join(layers_list),
        }
        if node_mode == "features":
            su = nodes.get(u, {})
            sv = nodes.get(v, {})
            row["source_compound"] = su.get("compound", "") or ""
            row["target_compound"] = sv.get("compound", "") or ""
        rows.append(row)

    cols = ["source", "target", "weight", "layer", "layers"]
    if node_mode == "features":
        cols += ["source_compound", "target_compound"]
    return pd.DataFrame(rows, columns=cols)

def _louvain_labels(
    G: nx.Graph,
    *,
    resolution: float = 1.0,
    weight: str = "weight",
) -> Dict[str, str]:
    if G.number_of_nodes() == 0:
        return {}

    labels: Dict[str, str] = {}
    module_idx = 1
    for comp_nodes in sorted(nx.connected_components(G), key=lambda c: (-len(c), sorted(map(str, c)))):
        H = G.subgraph(comp_nodes).copy()
        if H.number_of_edges() == 0:
            for node in sorted(map(str, H.nodes())):
                labels[node] = f"Module_{module_idx}"
                module_idx += 1
            continue
        total_weight = sum(float(d.get(weight, 1.0)) for _, _, d in H.edges(data=True))
        if total_weight <= 0:
            for node in sorted(map(str, H.nodes())):
                labels[node] = f"Module_{module_idx}"
            module_idx += 1
            continue

        comms = louvain_communities(H, weight=weight, resolution=float(resolution), seed=0)
        comms = sorted(comms, key=lambda c: (-len(c), sorted(map(str, c))))
        for comm in comms:
            for node in sorted(map(str, comm)):
                labels[node] = f"Module_{module_idx}"
            module_idx += 1
    return labels

def _connected_component_labels(G: nx.Graph) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    label = 2
    for comp in nx.connected_components(G):
        members = list(comp)
        if len(members) == 1:
            labels[str(members[0])] = "Group_I"
            continue
        name = f"Group_{label}"
        for node in members:
            labels[str(node)] = name
        label += 1
    return labels

def _graph_to_adjacency(ids: Sequence[str], G: nx.Graph) -> np.ndarray:
    idx = {str(node_id): i for i, node_id in enumerate(map(str, ids))}
    adj = np.zeros((len(idx), len(idx)), dtype=bool)
    for u, v in G.edges():
        i = idx[str(u)]
        j = idx[str(v)]
        adj[i, j] = True
        adj[j, i] = True
    return adj

def _feature_meta_lookup(names: Sequence[str], rodins: Sequence[object]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for name, rodin in zip(names, rodins):
        tag = (str(name) or "layer").replace(".", "_")
        uns = getattr(rodin, "uns", None)
        if isinstance(uns, dict):
            file_type = str(uns.get("file_type", ""))
        elif uns is None:
            file_type = ""
        else:
            file_type = str(getattr(uns, "file_type", ""))
        labels: Dict[str, str] = {}
        F = getattr(rodin, "features", None)
        if F is not None and not F.empty:
            F = _ensure_df(F, "r.features").copy()
            F.index = F.index.astype(str)
            lead = 2 if file_type.lower() == "metabolomics" else 1
            for fid in F.index:
                vals = [
                    str(v).strip()
                    for v in F.loc[fid].iloc[: min(lead, F.shape[1])].tolist()
                    if pd.notna(v) and str(v).strip() not in {"", "nan", "None"}
                ]
                labels[str(fid)] = "_".join(vals) if vals else str(fid)
        lookup[tag] = {"type": file_type, "labels": labels}
    return lookup

def _finalize_graph_collection(
    graphs: Dict[str, nx.Graph],
    *,
    node_mode: str,
    names: Sequence[str],
    rodins: Sequence[object],
    community_res: float,
    feature_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    feature_meta = feature_meta if node_mode == "features" and feature_meta is not None else (
        _feature_meta_lookup(names, rodins) if node_mode == "features" else {}
    )

    for G in graphs.values():
        G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_edges() > MAX_EDGES:
            warnings.warn(f"Network too dense: {G.number_of_edges()} edges (> {MAX_EDGES}).")
        for node in G.nodes():
            node_id = str(node)
            if node_mode == "features" and "__" in node_id:
                tag, fid = node_id.split("__", 1)
                meta = feature_meta.get(tag, {})
                if meta.get("type"):
                    G.nodes[node]["type"] = meta["type"]
                label = (meta.get("labels") or {}).get(fid)
                if label:
                    G.nodes[node]["compound"] = label

    def apply(prefix: str, default: str, parts: Dict[str, Dict[str, str]]) -> None:
        if len(parts) == 1:
            labels = next(iter(parts.values()))
            for G in graphs.values():
                for node in G.nodes():
                    G.nodes[node][prefix] = labels.get(str(node), default)
            return
        keys = {
            name: (
                f"{prefix}_entire" if name == "entire"
                else f"{prefix}_fused" if name == "fused"
                else f"{prefix}_con" if name == "consensus"
                else f"{prefix}_{re.sub(r'[^0-9A-Za-z_]+', '_', str(name)).strip('_').lower() or 'layer'}"
            )
            for name in parts
        }
        for graph_name, G in graphs.items():
            active_key = keys[graph_name]
            for node in G.nodes():
                node_id = str(node)
                for name, labels in parts.items():
                    G.nodes[node][keys[name]] = labels.get(node_id, default)
                G.nodes[node][prefix] = G.nodes[node][active_key]

    apply("community", "Group_I", {name: _connected_component_labels(Gx) for name, Gx in graphs.items()})
    apply("module", "Module_0", {name: _louvain_labels(Gx, resolution=community_res) for name, Gx in graphs.items()})

    return {name: _network_stats(Gx) for name, Gx in graphs.items()}

def _refresh_graph_entries(
    nt: "Netan",
    graphs: Dict[str, nx.Graph],
    infos: Dict[str, Dict[str, Any]],
    *,
    node_mode: str,
    community_res: float,
) -> None:
    feature_meta = None
    if node_mode == "features":
        feature_meta = nt._cache.get("feature_meta")
        if not isinstance(feature_meta, dict) or not feature_meta:
            feature_meta = _feature_meta_lookup(nt.names, nt.rodins)
            nt._cache["feature_meta"] = feature_meta
    stats_by_graph = _finalize_graph_collection(
        graphs,
        node_mode=node_mode,
        names=nt.names,
        rodins=nt.rodins,
        community_res=community_res,
        feature_meta=feature_meta,
    )
    for name, graph_obj in graphs.items():
        info_obj = dict(infos.get(name, nt._graph_info(name)))
        info_obj["community_res"] = float(community_res)
        nt._set_graph_entry(
            name,
            graph=graph_obj,
            info=info_obj,
            stats=stats_by_graph.get(name, {}),
        )

def _build_named_layer_graphs(
    items: Sequence[Tuple[str, pd.DataFrame, Sequence[str]]],
    *,
    method: str,
    n_jobs: int,
    infer_kwargs: Dict[str, Any],
    sparsify_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, nx.Graph], Dict[str, Dict[str, Any]], Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    G_layers: Dict[str, nx.Graph] = {}
    graph_cache: Dict[str, Dict[str, Any]] = {}
    matrix_cache: Dict[str, Dict[str, np.ndarray]] = {}
    adj_by_layer: Dict[str, np.ndarray] = {}
    for name, df, ids in items:
        key = str(name)
        G_layer, mats, adj, info = _infer_graph_bundle(
            df,
            ids,
            method=method,
            n_jobs=n_jobs,
            infer_kwargs=infer_kwargs,
            sparsify_kwargs=sparsify_kwargs,
        )
        _set_single_layer_attrs(G_layer, key)
        G_layers[key] = G_layer
        graph_cache[key] = dict(info)
        matrix_cache[key] = mats
        adj_by_layer[key] = adj
    return G_layers, graph_cache, matrix_cache, adj_by_layer

def _build_outputs(
    rodins: Sequence[object],
    names: Sequence[str],
    *,
    node_mode: str,
    layer_mode: str,
    method: str,
    n_jobs: int,
    infer_kwargs: Dict[str, Any],
    sparsify_kwargs: Dict[str, Any],
    combine: str,
    min_layers: int,
) -> BuildState:
    if node_mode == "samples":
        ids = list(map(str, rodins[0].X.columns))
        if layer_mode == "stack":
            G_all, mats_all, adj_all, info_all = _infer_graph_bundle(
                pd.concat([r.X for r in rodins], axis=0),
                ids,
                method=method,
                n_jobs=n_jobs,
                infer_kwargs=infer_kwargs,
                sparsify_kwargs=sparsify_kwargs,
            )
            _set_single_layer_attrs(G_all, "entire")
            return _build_state(
                graphs={"entire": G_all},
                infos={"entire": dict(info_all)},
                matrices={"entire": mats_all},
                tag_map={},
            )

        G_layers, graph_cache, matrix_cache, adj_by_layer = _build_named_layer_graphs(
            [(str(name), rodin.X, ids) for name, rodin in zip(names, rodins)],
            method=method,
            n_jobs=n_jobs,
            infer_kwargs=infer_kwargs,
            sparsify_kwargs=sparsify_kwargs,
        )
        derived = _samples_multilayer_derived(
            ids,
            matrix_cache,
            graph_cache,
            adj_by_layer,
            combine=combine,
            min_layers=min_layers,
            sparsify_kwargs=sparsify_kwargs,
        )
        matrix_cache["fused"] = matrix_cache["entire"] = derived["fused_mats"]
        graph_cache.update(derived["infos"])
        return _build_state(
            graphs={"entire": derived["graphs"]["entire"], "fused": derived["graphs"]["fused"], "consensus": derived["graphs"]["consensus"], **G_layers},
            infos=graph_cache,
            matrices=matrix_cache,
            tag_map={},
        )

    tagged, tag_map, node_to_layer = _tag_feature_frames(names, rodins)
    if layer_mode == "stack":
        tagged_all = pd.concat(tagged, axis=0)
        ids = list(map(str, tagged_all.index.astype(str)))
        G_all, mats_all, adj_all, info_all = _infer_graph_bundle(
            tagged_all.T,
            ids,
            method=method,
            n_jobs=n_jobs,
            infer_kwargs=infer_kwargs,
            sparsify_kwargs=sparsify_kwargs,
        )
        _annotate_feature_layers(G_all, node_to_layer, graph_label="entire")
        return _build_state(
            graphs={"entire": G_all},
            infos={"entire": dict(info_all)},
            matrices={"entire": mats_all},
            tag_map=tag_map,
        )

    G_layers, graph_cache, matrix_cache, adj_by_layer = _build_named_layer_graphs(
        [
            (str(name), Xtag.T, list(map(str, Xtag.index.astype(str))))
            for name, Xtag in zip(names, tagged)
        ],
        method=method,
        n_jobs=n_jobs,
        infer_kwargs=infer_kwargs,
        sparsify_kwargs=sparsify_kwargs,
    )
    tagged_all = pd.concat(tagged, axis=0)
    ids, cross_raw, cross_norm = _infer_feature_cross_matrix(
        tagged,
        method=method,
        n_jobs=n_jobs,
        infer_kwargs=infer_kwargs,
    )
    W_all_raw = _dispatch_infer(method, tagged_all.T, n_jobs, **infer_kwargs)
    mats_all = _matrix_cache_entry(W_all_raw, _normalize_matrix(W_all_raw))
    adj_cross, info_cross = _sparsify_matrix(cross_raw, cross_norm, **sparsify_kwargs)
    G_cross = _matrix_to_graph(ids, cross_raw, adj_cross)
    _set_single_layer_attrs(G_cross, "cross")
    G_all, graph_cache["entire"] = _feature_entire_graph(
        ids,
        G_layers,
        G_cross,
        W_abs=mats_all["W_abs"],
        W_norm=mats_all["W_norm"],
        base_info=_support_base_info([*graph_cache.values(), info_cross], fallback=info_cross),
    )
    graph_cache["cross"] = dict(info_cross)
    matrix_cache["entire"] = mats_all
    matrix_cache["cross"] = _matrix_cache_entry(cross_raw, cross_norm)
    return _build_state(
        graphs={"entire": G_all, "cross": G_cross, **G_layers},
        infos=graph_cache,
        matrices=matrix_cache,
        tag_map=tag_map,
    )

def _samples_multilayer_derived(
    ids: Sequence[str],
    layer_matrices: Dict[str, Dict[str, np.ndarray]],
    layer_infos: Dict[str, Dict[str, Any]],
    adj_by_layer: Dict[str, np.ndarray],
    *,
    combine: str,
    min_layers: int,
    sparsify_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    W_raw = {name: mats["W_raw"] for name, mats in layer_matrices.items()}
    W_abs = {name: mats["W_abs"] for name, mats in layer_matrices.items()}
    W_norm = {name: mats["W_norm"] for name, mats in layer_matrices.items()}
    fused_raw = _combine_matrix_stack(list(W_raw.values()), how=combine)
    fused_abs = _combine_matrix_stack(list(W_abs.values()), how=combine)
    fused_norm = _combine_matrix_stack(list(W_norm.values()), how=combine)
    fused_adj, fused_info = _sparsify_matrix(fused_abs, fused_norm, **sparsify_kwargs)
    fused_mats = _matrix_cache_entry(fused_raw, fused_norm, W_abs=fused_abs)
    fused_info = {**fused_info, "combine": combine}
    support_base = _support_base_info(list(layer_infos.values()), fallback=fused_info)

    G_fused = _matrix_to_graph(ids, fused_abs, fused_adj)
    _annotate_support_layers(G_fused, ids, adj_by_layer, graph_label="fused")

    G_all, adj_all = _build_consensus_graph(ids, W_abs, adj_by_layer, min_layers=1, combine=combine)
    _annotate_support_layers(G_all, ids, adj_by_layer, graph_label="entire")

    G_con, adj_con = _build_consensus_graph(ids, W_abs, adj_by_layer, min_layers=min_layers, combine=combine)
    _annotate_support_layers(G_con, ids, adj_by_layer, graph_label="consensus")

    return {
        "graphs": {"fused": G_fused, "entire": G_all, "consensus": G_con},
        "infos": {
            "fused": fused_info,
            "entire": _derived_graph_info(adj_all, base=support_base, W_abs=fused_abs, W_norm=fused_norm, combine=combine),
            "consensus": _derived_graph_info(adj_con, base=support_base, W_abs=fused_abs, W_norm=fused_norm, min_layers=min_layers, combine=combine),
        },
        "fused_mats": fused_mats,
    }

def _feature_entire_graph(
    ids: Sequence[str],
    layer_graphs: Dict[str, nx.Graph],
    G_cross: nx.Graph,
    *,
    W_abs: np.ndarray,
    W_norm: np.ndarray,
    base_info: Dict[str, Any],
) -> Tuple[nx.Graph, Dict[str, Any]]:
    G_all = _union_graphs_with_label([*layer_graphs.values(), G_cross], graph_label="entire")
    adj_all = _graph_to_adjacency(ids, G_all)
    return G_all, _derived_graph_info(adj_all, base=base_info, W_abs=W_abs, W_norm=W_norm)

def _requested_k(info: Dict[str, Any]):
    if info.get("k_mode") == "auto":
        return "auto"
    if info.get("k_mode") == "none":
        return None
    return info.get("k")

def _display_k(info: Dict[str, Any]) -> str:
    req = info.get("k_input", _requested_k(info))
    eff = info.get("k")
    if req == "auto":
        return f"auto({eff})" if eff is not None else "auto(off:sparse)"
    return "None" if req is None else str(req)

def _extract_sample_ids_from_df(samples_df: pd.DataFrame, expected_ids: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    if not isinstance(samples_df, pd.DataFrame) or samples_df.empty:
        return None
    expected = None if expected_ids is None else set(map(str, expected_ids))

    def _match(values: pd.Series) -> Optional[List[str]]:
        vals = values.astype(str).tolist()
        if not values.is_unique:
            return None
        if expected is not None and set(vals) != expected:
            return None
        return vals

    cols = list(samples_df.columns)
    ordered_cols: List[Any] = []
    ordered_cols.extend([col for col in cols if str(col).strip().lower() == "id"])
    if cols and cols[0] not in ordered_cols:
        ordered_cols.append(cols[0])
    ordered_cols.extend([col for col in cols if col not in ordered_cols])

    for col in ordered_cols:
        matched = _match(samples_df[col])
        if matched is not None:
            return matched
    if samples_df.index.is_unique:
        matched = _match(pd.Series(samples_df.index, index=samples_df.index))
        if matched is not None:
            return matched
    return None

def _validate_layer_names(names: Sequence[str]) -> List[str]:
    cleaned = list(map(str, names))
    reserved = {name.lower() for name in _CORE_GRAPHS}
    bad = [name for name in cleaned if name.lower() in reserved]
    if bad:
        raise ValueError(
            "Layer names cannot use reserved graph names "
            f"{sorted(_CORE_GRAPHS)}. Got: {bad}"
        )
    seen, dup = set(), []
    for name in cleaned:
        key = name.lower()
        if key in seen and name not in dup:
            dup.append(name)
        seen.add(key)
    if dup:
        raise ValueError(f"Layer names must be unique (case-insensitive). Duplicates: {dup}")
    return cleaned


def _dispatch_infer(
    method: str,
    df: pd.DataFrame,
    n_jobs: int,
    **kwargs,
):
    method = method.lower()

    if method == "clr":
        return _clr_matrix(df, n_jobs=n_jobs, n_neighbors=int(kwargs.get("n_neighbors", 2)))
    if method == "rf":
        md = kwargs.get("max_depth", None)
        return _rf_matrix(
            df,
            n_jobs=n_jobs,
            n_estimators=int(kwargs.get("n_estimators", 160)),
            max_depth=(None if md in (None, "", 0, "0") else int(md)),
        )
    if method == "glasso":
        return _glasso_matrix(
            df,
            alpha=float(kwargs.get("alpha", 0.05)),
            max_iter=int(kwargs.get("max_iter", 200)),
            tol=float(kwargs.get("tol", 1e-4)),
        )
    return _corr_matrix(df)

def common_samples(objs) -> List[str]:
    """
    Return sample IDs shared by all inputs.

    Parameters
    ----------
    objs : sequence
        Rodin-like objects exposing sample IDs in ``X.columns``.

    Returns
    -------
    list[str]
        Shared sample IDs in the order of the first input.
    """
    objs = [objs] if not isinstance(objs, (list, tuple)) else list(objs)
    if not objs:
        raise ValueError("Provide at least one object.")
    first = list(map(str, objs[0].X.columns))
    common = set.intersection(*(set(map(str, r.X.columns)) for r in objs))
    seen = set()
    return [s for s in first if s in common and not (s in seen or seen.add(s))]

def _subset_rodin(r, *, features: Optional[Sequence[str]] = None, samples: Optional[Sequence[str]] = None):
    X = _ensure_df(getattr(r, "X", None), "r.X").copy()
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    feat_ids = list(X.index) if features is None else list(map(str, features))
    sample_ids = list(X.columns) if samples is None else list(map(str, samples))
    X_sub = X.loc[feat_ids, sample_ids]
    try:
        out = r[X_sub]
    except Exception:
        out = copy.deepcopy(r)
        out.X = X_sub
        F = getattr(out, "features", None)
        if isinstance(F, pd.DataFrame):
            F = F.copy()
            F.index = F.index.astype(str)
            out.features = F.loc[[f for f in feat_ids if f in F.index]]
        S = getattr(out, "samples", None)
        if isinstance(S, pd.DataFrame) and not S.empty:
            S = S.copy()
            detected_ids = _extract_sample_ids_from_df(S)
            if detected_ids is not None:
                keep = [s for s in sample_ids if s in set(detected_ids)]
                aligned = S.assign(_sample_id=list(map(str, detected_ids))).set_index("_sample_id").loc[keep].reset_index()
                if "id" in aligned.columns:
                    aligned["id"] = aligned["_sample_id"].astype(str)
                    aligned = aligned.drop(columns=["_sample_id"])
                elif len(aligned.columns) > 1 and aligned.iloc[:, 1].astype(str).tolist() == aligned["_sample_id"].astype(str).tolist():
                    aligned = aligned.drop(columns=["_sample_id"])
                else:
                    aligned = aligned.rename(columns={"_sample_id": "id"})
                out.samples = aligned
    S = getattr(out, "samples", None)
    if isinstance(S, pd.DataFrame):
        out.samples.reset_index(drop=True, inplace=True)
    return out

def _derived_graph_info(
    adj: np.ndarray,
    *,
    base: Dict[str, Any],
    W_abs: np.ndarray,
    W_norm: np.ndarray,
    min_layers: Optional[int] = None,
    combine: Optional[str] = None,
) -> Dict[str, Any]:
    thr_raw_eff, thr_norm_eff = _effective_threshold_pair(
        W_abs,
        W_norm,
        adj,
        thr_raw_input=base.get("thr_raw_input"),
        thr_norm_input=base.get("thr_norm_input", base.get("thr_norm")),
    )
    info = {
        "thr_raw": thr_raw_eff,
        "thr_norm": thr_norm_eff,
        "thr_raw_input": base.get("thr_raw_input"),
        "thr_norm_input": base.get("thr_norm_input"),
        "auto_info": base.get("auto_info"),
        "auto_target": base.get("auto_target"),
        "attach_isolates": bool(base.get("attach_isolates", False)),
        "k": base.get("k"),
        "k_mode": base.get("k_mode"),
        "mutual": bool(base.get("mutual", False)),
        "isolate_count": int(np.count_nonzero(~adj.any(axis=1))),
        "isolates_reattached": 0,
        "used_isolate_reattachment": False,
    }
    if min_layers is not None:
        info["min_layers"] = int(min_layers)
    if combine is not None:
        info["combine"] = str(combine)
    return info
