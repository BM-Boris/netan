"""Internal ranking helpers for Netan."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from . import netan as _n


def _bh_fdr_impl(pvals: Sequence[float]) -> np.ndarray:
    vals = np.asarray(pvals, dtype=np.float64)
    out = np.full(vals.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return out
    p = vals[mask]
    order = np.argsort(p)
    ranked = p[order]
    n = ranked.size
    q = ranked * n / np.arange(1, n + 1, dtype=np.float64)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0.0, 1.0)
    out[mask] = restored
    return out


def _edge_partition_impl(
    class_index: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, Any]:
    ci = class_index[edge_i]
    cj = class_index[edge_j]
    same_mask = ci == cj
    within_idx = np.flatnonzero(same_mask)
    cross_idx = np.flatnonzero(~same_mask)
    n_classes = int(class_index.max()) + 1 if class_index.size else 0
    class_sizes = np.bincount(class_index, minlength=n_classes).astype(np.float64) if n_classes else np.zeros(0, dtype=np.float64)
    within_parts: List[Tuple[np.ndarray, float]] = []
    for c in range(n_classes):
        idx = np.flatnonzero((ci == c) & (cj == c))
        within_parts.append((idx, float(weights[idx].sum())))

    pair_parts: List[Dict[str, Any]] = []
    beta_sum = 0.0
    for c in range(n_classes):
        for d in range(c + 1, n_classes):
            idx = np.flatnonzero(((ci == c) & (cj == d)) | ((ci == d) & (cj == c)))
            den = float(weights[idx].sum())
            if den <= 0 or within_parts[c][1] <= 0 or within_parts[d][1] <= 0:
                continue
            support = float(min(den, within_parts[c][1], within_parts[d][1]))
            beta = float(
                np.sqrt(max(class_sizes[c], 1.0) * max(class_sizes[d], 1.0))
                * np.sqrt(max(support, 1e-12))
            )
            pair_parts.append({"c": c, "d": d, "idx": idx, "den": den, "beta": beta})
            beta_sum += beta

    class_alpha = np.zeros(n_classes, dtype=np.float64)
    if beta_sum > 0:
        for part in pair_parts:
            part["beta"] /= beta_sum
            class_alpha[part["c"]] += 0.5 * part["beta"]
            class_alpha[part["d"]] += 0.5 * part["beta"]
    return {
        "same_edges": int(np.count_nonzero(same_mask)),
        "cross_edges": int(np.count_nonzero(~same_mask)),
        "within_idx": within_idx,
        "cross_idx": cross_idx,
        "within_den": float(weights[within_idx].sum()),
        "cross_den": float(weights[cross_idx].sum()),
        "within_parts": within_parts,
        "pair_parts": pair_parts,
        "class_alpha": class_alpha,
    }


def _edge_feature_score_impl(
    weighted: np.ndarray,
    partition: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = int(weighted.shape[0])
    zeros = np.zeros(n_features, dtype=np.float32)
    within_parts = partition.get("within_parts") or []
    pair_parts = partition.get("pair_parts") or []
    class_alpha_raw = partition.get("class_alpha")
    class_alpha = np.asarray(class_alpha_raw if class_alpha_raw is not None else [], dtype=np.float64)

    if pair_parts:
        within_by_class: List[np.ndarray] = []
        for idx, den in within_parts:
            arr = (
                (weighted[:, idx].sum(axis=1) / max(float(den), 1e-12)).astype(np.float32, copy=False)
                if idx.size and float(den) > 0
                else zeros
            )
            within_by_class.append(arr)

        within = np.zeros(n_features, dtype=np.float32)
        between = np.zeros(n_features, dtype=np.float32)
        score = np.zeros(n_features, dtype=np.float32)
        for part in pair_parts:
            c = int(part["c"])
            d = int(part["d"])
            idx = part["idx"]
            den = float(part["den"])
            beta = float(part["beta"])
            b_cd = (
                (weighted[:, idx].sum(axis=1) / max(den, 1e-12)).astype(np.float32, copy=False)
                if idx.size and den > 0
                else zeros
            )
            between += np.float32(beta) * b_cd
            score += np.float32(beta) * (
                np.log1p(b_cd) - 0.5 * (np.log1p(within_by_class[c]) + np.log1p(within_by_class[d]))
            ).astype(np.float32, copy=False)

        if class_alpha.size:
            for c, alpha in enumerate(class_alpha):
                if alpha > 0:
                    within += np.float32(alpha) * within_by_class[c]
        return within, between, score

    within_idx = partition["within_idx"]
    cross_idx = partition["cross_idx"]
    within = (
        (weighted[:, within_idx].sum(axis=1) / max(float(partition["within_den"]), 1e-12)).astype(np.float32, copy=False)
        if within_idx.size and float(partition["within_den"]) > 0
        else zeros
    )
    between = (
        (weighted[:, cross_idx].sum(axis=1) / max(float(partition["cross_den"]), 1e-12)).astype(np.float32, copy=False)
        if cross_idx.size and float(partition["cross_den"]) > 0
        else zeros
    )
    score = np.log1p(between) - np.log1p(within)
    return within, between, score.astype(np.float32, copy=False)


def _rank_result_impl(
    nt: "Any",
    *,
    label: str,
    G: nx.Graph,
    graph_name: str,
    label_map: Dict[str, Any],
    layers: Optional[Union[str, Sequence[str]]],
    use_weights: bool,
    standardize: bool,
    n_perm: int,
    seed: int,
    fdr: bool,
    chunk_size: int,
) -> Dict[str, Any]:
    graph_nodes = list(map(str, G.nodes()))
    if len(graph_nodes) < 2 or G.number_of_edges() == 0:
        raise ValueError("rank() requires a graph with at least one edge.")

    node_idx = {node: i for i, node in enumerate(graph_nodes)}
    labels_full = np.asarray([label_map.get(node, None) for node in graph_nodes], dtype=object)
    labeled_nodes = np.flatnonzero(np.asarray([lab is not None for lab in labels_full], dtype=bool))
    if labeled_nodes.size < 2:
        raise ValueError("label column does not label enough graph nodes.")

    compact = np.full(len(graph_nodes), -1, dtype=np.int32)
    compact[labeled_nodes] = np.arange(labeled_nodes.size, dtype=np.int32)
    edge_i, edge_j, edge_w = [], [], []
    for u, v, data in G.edges(data=True):
        iu = compact[node_idx[str(u)]]
        iv = compact[node_idx[str(v)]]
        if iu < 0 or iv < 0:
            continue
        edge_i.append(iu)
        edge_j.append(iv)
        edge_w.append(float(data.get("weight", 1.0) or 0.0))
    if not edge_i:
        raise ValueError("rank() found no labeled edges in the selected graph.")

    edge_i = np.asarray(edge_i, dtype=np.int32)
    edge_j = np.asarray(edge_j, dtype=np.int32)
    weights = np.asarray(edge_w, dtype=np.float32)
    if not use_weights:
        weights = np.ones_like(weights, dtype=np.float32)

    labels = labels_full[labeled_nodes]
    class_labels, class_index = np.unique(labels, return_inverse=True)
    partition = _edge_partition_impl(class_index, edge_i, edge_j, weights)
    same_edges = int(partition["same_edges"])
    cross_edges = int(partition["cross_edges"])
    if same_edges == 0 or cross_edges == 0:
        raise ValueError("rank() requires both same-label and cross-label labeled edges.")
    uses_pairwise = bool(partition.get("pair_parts"))

    label_to_pos = {cls: idx for idx, cls in enumerate(class_labels)}
    perm_parts: List[Dict[str, Any]] = []
    if int(n_perm) > 0:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_perm)):
            perm_index = np.asarray([label_to_pos[val] for val in rng.permutation(labels)], dtype=np.int32)
            part = _edge_partition_impl(perm_index, edge_i, edge_j, weights)
            if part["same_edges"] > 0 and part["cross_edges"] > 0:
                perm_parts.append(part)

    selected = list(map(str, nt.names)) if layers is None else []
    if layers is not None:
        wanted = [str(x) for x in _n._grid_list(layers)]
        lookup = {str(name).lower(): str(name) for name in nt.names}
        selected = []
        for name in wanted:
            key = lookup.get(name.lower())
            if key is None:
                raise ValueError(f"Unknown layer '{name}'. Available layers: {list(map(str, nt.names))}")
            if key not in selected:
                selected.append(key)

    feature_meta = nt._cache.get("feature_meta")
    if not isinstance(feature_meta, dict) or not feature_meta:
        feature_meta = _n._feature_meta_lookup(nt.names, nt.rodins)
        nt._cache["feature_meta"] = feature_meta
    rows: List[pd.DataFrame] = []
    ordered_cols = [graph_nodes[i] for i in labeled_nodes]
    for layer_name, rodin in zip(map(str, nt.names), nt.rodins):
        if layer_name not in selected:
            continue
        tag = (layer_name or "layer").replace(".", "_")
        feature_labels = (feature_meta.get(tag) or {}).get("labels", {})
        Xdf = _n._ensure_df(getattr(rodin, "X", None), "r.X")
        Xdf = Xdf.loc[:, ordered_cols]
        feats = Xdf.index.astype(str).tolist()
        display_feats = [feature_labels.get(fid, fid) for fid in feats]
        X = Xdf.to_numpy(dtype=np.float32, copy=True)
        valid = np.isfinite(X)
        counts = valid.sum(axis=1, keepdims=True)
        means = np.divide(np.where(valid, X, 0.0).sum(axis=1, keepdims=True), np.maximum(counts, 1), dtype=np.float32)
        X = np.where(valid, X, means)
        if standardize:
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            sd[sd == 0] = 1.0
            X = (X - mu) / sd

        within_all = np.zeros(X.shape[0], dtype=np.float32)
        between_all = np.zeros(X.shape[0], dtype=np.float32)
        score_all = np.zeros(X.shape[0], dtype=np.float32)
        p_all = np.full(X.shape[0], np.nan, dtype=np.float32)
        class_means = np.column_stack([X[:, class_index == i].mean(axis=1) for i in range(len(class_labels))]).astype(np.float32)
        best_idx = np.argmax(class_means, axis=1)
        if class_means.shape[1] > 1:
            second = np.partition(class_means, -2, axis=1)[:, -2]
        else:
            second = np.zeros(class_means.shape[0], dtype=np.float32)
        dominant_class = np.asarray(class_labels, dtype=object)[best_idx]
        class_margin = class_means[np.arange(class_means.shape[0]), best_idx] - second

        for start in range(0, X.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), X.shape[0])
            diff2 = (X[start:stop, edge_i] - X[start:stop, edge_j]) ** 2
            weighted = diff2 * weights[None, :]
            within, between, score = _edge_feature_score_impl(weighted, partition)
            within_all[start:stop] = within
            between_all[start:stop] = between
            score_all[start:stop] = score
            if perm_parts:
                ge = np.ones(stop - start, dtype=np.int32)
                for perm_part in perm_parts:
                    perm_score = _edge_feature_score_impl(weighted, perm_part)[2]
                    ge += perm_score >= score
                p_all[start:stop] = ge / float(len(perm_parts) + 1)

        rows.append(
            pd.DataFrame(
                {
                    "feature_id": feats,
                    "feature": display_feats,
                    "layer": layer_name,
                    "graph": graph_name,
                    "label": str(label),
                    "score": score_all.astype(np.float64),
                    "between_dispersion": between_all.astype(np.float64),
                    "within_dispersion": within_all.astype(np.float64),
                    "mean_cross_diff": np.sqrt(np.maximum(between_all, 0)).astype(np.float64),
                    "mean_same_diff": np.sqrt(np.maximum(within_all, 0)).astype(np.float64),
                    "dominant_class": dominant_class.astype(str),
                    "class_margin": class_margin.astype(np.float64),
                    "n_labeled_samples": labeled_nodes.size,
                    "same_edges": same_edges,
                    "cross_edges": cross_edges,
                    "p_perm": p_all.astype(np.float64),
                }
            )
        )

    if not rows:
        raise ValueError("No layers were selected for ranking.")

    details = pd.concat(rows, axis=0, ignore_index=True)
    details["p_adj"] = _bh_fdr_impl(details["p_perm"].to_numpy(dtype=np.float64)) if fdr and int(n_perm) > 0 and perm_parts else np.nan
    details = details.sort_values(["score", "p_perm"], ascending=[False, True], na_position="last").reset_index(drop=True)
    details["rank"] = np.arange(1, len(details) + 1, dtype=int)
    details = _n._feature_identity_index(details)
    out = details[["rank", "feature_id", "feature", "layer", "dominant_class", "score", "p_perm", "p_adj"]].rename(
        columns={"dominant_class": "top_class"}
    )
    out = _n._feature_identity_index(out)
    return {
        "graph": graph_name,
        "label": str(label),
        "layers": selected,
        "use_weights": bool(use_weights),
        "standardize": bool(standardize),
        "n_perm": int(n_perm),
        "valid_perm": len(perm_parts),
        "contrast_mode": ("pairwise" if uses_pairwise else "pooled"),
        "same_edges": same_edges,
        "cross_edges": cross_edges,
        "table": out,
        "details": details,
    }
