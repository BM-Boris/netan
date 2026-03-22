"""Internal scoring and tuning helpers for Netan."""

import copy
import warnings
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm.auto import tqdm

from . import netan as _n
from ._build import _requested_k
from ._views import (
    _aligned_samples_df,
    _fmt_method_params,
    _graph_semantics,
    _mask_inactive_threshold_fields,
    _public_param_state,
    _silent,
)


def _sample_label_map_impl(nt: "Any", column: str) -> Dict[str, Any]:
    cache = nt._cache.setdefault("label_maps", {})
    if column in cache:
        return dict(cache[column])
    samples_df = _aligned_samples_df(nt)
    if column not in samples_df.columns:
        raise ValueError(f"label column '{column}' was not found in r.samples.")
    label_map = {
        str(sample_id): label
        for sample_id, label in zip(nt.sample_ids, samples_df[column].tolist())
        if pd.notna(label)
    }
    if len(set(label_map.values())) < 2:
        raise ValueError("label column must contain at least two classes.")
    cache[column] = dict(label_map)
    return label_map


def _graph_modularity01_impl(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    groups: Dict[str, set] = {}
    for node, data in G.nodes(data=True):
        groups.setdefault(str(data.get("module", "Module_0")), set()).add(node)
    part = list(groups.values())
    if len(part) <= 1:
        return 0.0
    try:
        return float(np.clip(nx_modularity(G, part, weight="weight"), 0.0, 1.0))
    except Exception:
        return 0.0


def _degree_band_score_impl(mean_degree_active: float) -> float:
    deg = float(mean_degree_active)
    if not np.isfinite(deg) or deg <= 1.0:
        return 0.0

    def smoothstep01(x: float) -> float:
        z = float(np.clip(x, 0.0, 1.0))
        return z * z * (3.0 - 2.0 * z)

    if deg < 4.0:
        return float(0.95 * smoothstep01((deg - 1.0) / 3.0))
    if deg < 8.0:
        return float(0.95 + 0.05 * smoothstep01((deg - 4.0) / 4.0))
    if deg <= 12.0:
        return float(0.95 + 0.05 * smoothstep01((12.0 - deg) / 4.0))
    if deg < 36.0:
        return float(0.95 * smoothstep01((36.0 - deg) / 24.0))
    return 0.0


def _module_size_band_score_impl(active_nodes: int, modules: int) -> float:
    a = float(active_nodes)
    m = float(modules)
    if not np.isfinite(a) or not np.isfinite(m) or a <= 0.0 or m <= 0.0:
        return 0.0

    avg = a / m

    def smoothstep01(x: float) -> float:
        z = float(np.clip(x, 0.0, 1.0))
        return z * z * (3.0 - 2.0 * z)

    if avg <= 1.5:
        return 0.0
    if avg < 4.0:
        return float(smoothstep01((avg - 1.5) / 2.5))
    if avg <= 10.0:
        return 1.0
    if avg < 18.0:
        return float(1.0 - 0.15 * smoothstep01((avg - 10.0) / 8.0))
    if avg < 28.0:
        return float(0.85 * (1.0 - smoothstep01((avg - 18.0) / 10.0)))
    return 0.0


_DEFAULT_SCORE_WEIGHTS_IMPL = {
    "structure_supervised": {"modularity01": 20.0, "degree_band": 20.0, "module_size_band": 60.0},
    "structure_unsupervised": {"modularity01": 20.0, "degree_band": 20.0, "module_size_band": 60.0},
    "stab_supervised": {"module_stability": 30.0, "edge_stability": 70.0},
    "stab_unsupervised": {"module_stability": 30.0, "edge_stability": 70.0},
    "sep": {"ari01": 15.0, "nmi": 5.0, "assort01": 80.0},
    "supervised": {"sep": 60.0, "structure": 15.0, "stab": 15.0, "active_fraction": 10.0},
    "unsupervised": {"structure": 50.0, "stab": 40.0, "active_fraction": 10.0},
}

def _normalize_weight_block_impl(
    value: Optional[Any],
    *,
    names: Sequence[str],
    defaults: Dict[str, float],
    block_name: str,
) -> Dict[str, float]:
    block = {str(name): float(defaults[str(name)]) for name in names}
    if value is None:
        pass
    elif isinstance(value, dict):
        unknown = sorted(set(map(str, value)) - set(names))
        if unknown:
            raise ValueError(f"Unknown keys in weights['{block_name}']: {unknown}")
        for name in names:
            if name in value:
                block[name] = float(value[name])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = list(value)
        if len(seq) != len(names):
            raise ValueError(f"weights['{block_name}'] must have length {len(names)}.")
        block = {str(name): float(val) for name, val in zip(names, seq)}
    else:
        raise TypeError(f"weights['{block_name}'] must be a dict or a sequence.")
    if any((not np.isfinite(val)) or float(val) < 0.0 for val in block.values()):
        raise ValueError(f"weights['{block_name}'] must contain finite non-negative values.")
    total = float(sum(block.values()))
    if total <= 0.0:
        raise ValueError(f"weights['{block_name}'] must sum to a positive value.")
    return {name: float(val / total) for name, val in block.items()}


def _resolve_score_weights_impl(weights: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
    if weights is not None and not isinstance(weights, dict):
        raise TypeError("weights must be a dict with blocks such as 'sep', 'structure_supervised', 'structure_unsupervised', 'stab_supervised', 'stab_unsupervised', 'supervised', and 'unsupervised'.")
    normalized = {} if weights is None else {str(key): val for key, val in weights.items()}
    unknown_top = sorted(set(normalized) - set(_DEFAULT_SCORE_WEIGHTS_IMPL))
    if unknown_top:
        raise ValueError(f"Unknown weight blocks: {unknown_top}")
    return {
        "structure_supervised": _normalize_weight_block_impl(
            normalized.get("structure_supervised"),
            names=("modularity01", "degree_band", "module_size_band"),
            defaults=_DEFAULT_SCORE_WEIGHTS_IMPL["structure_supervised"],
            block_name="structure_supervised",
        ),
        "structure_unsupervised": _normalize_weight_block_impl(
            normalized.get("structure_unsupervised"),
            names=("modularity01", "degree_band", "module_size_band"),
            defaults=_DEFAULT_SCORE_WEIGHTS_IMPL["structure_unsupervised"],
            block_name="structure_unsupervised",
        ),
        "stab_supervised": _normalize_weight_block_impl(
            normalized.get("stab_supervised"),
            names=("module_stability", "edge_stability"),
            defaults=_DEFAULT_SCORE_WEIGHTS_IMPL["stab_supervised"],
            block_name="stab_supervised",
        ),
        "stab_unsupervised": _normalize_weight_block_impl(
            normalized.get("stab_unsupervised"),
            names=("module_stability", "edge_stability"),
            defaults=_DEFAULT_SCORE_WEIGHTS_IMPL["stab_unsupervised"],
            block_name="stab_unsupervised",
        ),
        "sep": _normalize_weight_block_impl(
            normalized.get("sep"),
            names=("ari01", "nmi", "assort01"),
            defaults=_DEFAULT_SCORE_WEIGHTS_IMPL["sep"],
            block_name="sep",
        ),
        "supervised": _normalize_weight_block_impl(
            normalized.get("supervised"),
            names=("sep", "structure", "stab", "active_fraction"),
            defaults=_DEFAULT_SCORE_WEIGHTS_IMPL["supervised"],
            block_name="supervised",
        ),
        "unsupervised": _normalize_weight_block_impl(
            normalized.get("unsupervised"),
            names=("structure", "stab", "active_fraction"),
            defaults=_DEFAULT_SCORE_WEIGHTS_IMPL["unsupervised"],
            block_name="unsupervised",
        ),
    }

def _score_terms_impl(
    metrics: Dict[str, float],
    objective: str,
    *,
    weights: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    resolved = _resolve_score_weights_impl(weights)

    def z(name: str) -> float:
        return float(np.clip(float(metrics.get(name, 0.0)), 0.0, 1.0))

    active_fraction = z("active_fraction")
    if objective == "supervised":
        structure = sum(float(weight) * z(name) for name, weight in resolved["structure_supervised"].items())
        stab = sum(float(weight) * z(name) for name, weight in resolved["stab_supervised"].items())
        sep = sum(float(weight) * z(name) for name, weight in resolved["sep"].items())
        score = (
            float(resolved["supervised"]["sep"]) * sep
            + float(resolved["supervised"]["structure"]) * structure
            + float(resolved["supervised"]["stab"]) * stab
            + float(resolved["supervised"]["active_fraction"]) * active_fraction
        )
        return {
            "structure": float(np.clip(structure, 0.0, 1.0)),
            "sep": float(np.clip(sep, 0.0, 1.0)),
            "stab": float(np.clip(stab, 0.0, 1.0)),
            "score": float(np.clip(score, 0.0, 1.0)),
        }
    structure = sum(float(weight) * z(name) for name, weight in resolved["structure_unsupervised"].items())
    stab = sum(float(weight) * z(name) for name, weight in resolved["stab_unsupervised"].items())
    score = (
        float(resolved["unsupervised"]["structure"]) * structure
        + float(resolved["unsupervised"]["stab"]) * stab
        + float(resolved["unsupervised"]["active_fraction"]) * active_fraction
    )
    return {
        "structure": float(np.clip(structure, 0.0, 1.0)),
        "stab": float(np.clip(stab, 0.0, 1.0)),
        "score": float(np.clip(score, 0.0, 1.0)),
    }


def _graph_metrics_impl(
    nt: "Any",
    graph_name: str,
    *,
    objective: str,
    label_map: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    _, G, info = nt._graph_context(graph_name)
    stats = nt._graph_stats(graph_name)
    num_nodes = int(stats.get("numNodes", 0))
    active_nodes = int(stats.get("nodesWithEdges", 0))
    isolate_count = int(info.get("isolate_count", max(num_nodes - active_nodes, 0)))
    active_fraction = float(active_nodes / num_nodes) if num_nodes else 0.0
    isolate_fraction = float(isolate_count / num_nodes) if num_nodes else 0.0
    deg = np.asarray([float(d) for _, d in G.degree() if d > 0], dtype=np.float32)
    if deg.size == 0:
        mean_degree_active, median_degree_active, max_degree_active = 0.0, 0.0, 0.0
    else:
        mean_degree_active = float(deg.mean())
        median_degree_active = float(np.median(deg))
        max_degree_active = float(deg.max())
    module_map = {str(node): str(data.get("module", "Module_0")) for node, data in G.nodes(data=True)}
    metrics = {
        "nodes": num_nodes,
        "edges": int(stats.get("numEdges", 0)),
        "active_nodes": active_nodes,
        "isolates": isolate_count,
        "communities": int(stats.get("numCommunities", 0)),
        "modules": int(stats.get("numModules", 0)),
        "density_all": float(stats.get("densityAll", 0.0)),
        "density_active": float(stats.get("densityActive", 0.0)),
        "active_fraction": active_fraction,
        "isolate_fraction": isolate_fraction,
        "retain": 1.0 - isolate_fraction,
        "mean_degree_active": mean_degree_active,
        "median_degree_active": median_degree_active,
        "max_degree_active": max_degree_active,
        "degree_band": _degree_band_score_impl(mean_degree_active),
        "module_size_band": _module_size_band_score_impl(active_nodes, int(stats.get("numModules", 0))),
        "modularity01": _graph_modularity01_impl(G),
        "ari": 0.0,
        "ari01": 0.0,
        "nmi": 0.0,
        "label_assortativity": 0.0,
        "assort01": 0.0,
        "module_stability": 0.0,
        "edge_stability": 0.0,
        "_edge_set": {tuple(sorted((str(u), str(v)))) for u, v in G.edges()},
        "_module_map": module_map,
    }
    if objective == "supervised":
        if not label_map:
            raise ValueError("label_map is required for supervised tuning.")
        metrics.update(
            _supervised_terms_impl(
                module_map=module_map,
                edge_set=metrics["_edge_set"],
                label_map={str(node): label_map[str(node)] for node in G.nodes() if str(node) in label_map},
            )
        )
    return metrics


def _label_assortativity_from_edges_impl(edge_set: Any, label_map: Dict[str, Any]) -> float:
    if not edge_set:
        return 0.0
    H = nx.Graph()
    H.add_edges_from((str(u), str(v)) for u, v in edge_set)
    labeled = [node for node in H.nodes() if str(node) in label_map]
    if len(labeled) < 2:
        return 0.0
    H = H.subgraph(labeled).copy()
    if H.number_of_edges() == 0:
        return 0.0
    nx.set_node_attributes(H, {node: str(label_map[str(node)]) for node in H.nodes()}, "_label")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            score = float(nx.attribute_assortativity_coefficient(H, "_label"))
    except Exception:
        return 0.0
    return float(np.clip(score, -1.0, 1.0)) if np.isfinite(score) else 0.0


def _supervised_terms_impl(
    *,
    module_map: Dict[str, str],
    edge_set: Any,
    label_map: Dict[str, Any],
) -> Dict[str, float]:
    module_map = {str(node): str(module) for node, module in dict(module_map or {}).items()}
    shared = sorted(set(module_map) & set(label_map))
    if len(shared) < 2:
        raise ValueError("label column must contain at least two classes on the compared nodes.")
    labels = [label_map[node] for node in shared]
    if len(set(labels)) < 2:
        raise ValueError("label column must contain at least two classes on the compared nodes.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ari = float(adjusted_rand_score(labels, [module_map[node] for node in shared]))
        nmi = float(normalized_mutual_info_score(labels, [module_map[node] for node in shared]))
    assort = _label_assortativity_from_edges_impl(edge_set, label_map)
    return {
        "ari": ari,
        "ari01": float(np.clip(max(ari, 0.0), 0.0, 1.0)),
        "nmi": float(np.clip(nmi, 0.0, 1.0)),
        "label_assortativity": assort,
        "assort01": float(np.clip((assort + 1.0) / 2.0, 0.0, 1.0)),
    }


def _prepare_tune_configs_impl(
    *,
    node_mode: str,
    n_layers: int,
    layer_modes: Optional[Union[str, Sequence[str]]],
    graphs: Optional[Union[str, Sequence[str]]],
    methods: Optional[Sequence[str]],
    method_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]],
    combine: Optional[Union[str, Sequence[str]]],
    auto_target: Optional[Union[float, Sequence[float]]],
    thr_norm: Optional[Union[float, Sequence[float]]],
    thr_raw: Optional[Union[float, Sequence[float]]],
    k: Optional[Union[Any, Sequence[Any]]],
    mutual: Optional[Union[bool, Sequence[bool]]],
    attach_isolates: Optional[Union[bool, Sequence[bool]]],
    min_layers: Optional[Union[int, Sequence[int]]],
    community_res: float,
    n_jobs: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    single_layer = int(n_layers) < 2
    layer_mode_values = ["stack"] if (layer_modes is None and single_layer) else (
        ["stack", "multilayer"] if layer_modes is None else [str(x).lower() for x in _n._grid_list(layer_modes)]
    )
    if any(mode not in {"stack", "multilayer"} for mode in layer_mode_values):
        raise ValueError("layer_modes must contain only 'stack' and/or 'multilayer'.")
    if single_layer and "multilayer" in layer_mode_values:
        raise ValueError("tune(layer_modes='multilayer') requires at least two layers.")

    default_graphs = ["entire"] if single_layer else (["entire", "fused"] if node_mode == "samples" else ["entire"])
    graph_values = default_graphs if graphs is None else [str(x).lower() for x in _n._grid_list(graphs)]
    allowed_graphs = {"entire"} if single_layer else set(sum(_n._VALID_TUNE_GRAPHS[node_mode].values(), []))
    unknown_graphs = sorted({name for name in graph_values if name not in allowed_graphs})
    if unknown_graphs:
        raise ValueError(f"Unsupported tune graphs for node_mode='{node_mode}': {unknown_graphs}")

    method_grid_map = copy.deepcopy(_n._DEFAULT_TUNE_METHOD_GRIDS)
    if method_grids:
        for method_name, grid in method_grids.items():
            key = str(method_name).lower()
            method_grid_map[key] = {k: _n._grid_list(v) for k, v in (method_grid_map.get(key) or {}).items()}
            for param_name, values in (grid or {}).items():
                method_grid_map[key][str(param_name)] = _n._grid_list(values)
    method_values = ["spearman", "clr", "rf"] if methods is None else [str(x).lower() for x in _n._grid_list(methods)]
    if method_grids:
        for method in map(str.lower, method_grids):
            if method not in method_values:
                method_values.append(method)
    bad_methods = sorted({method for method in method_values if method not in _n.BUILDERS})
    if bad_methods:
        raise ValueError(f"Unknown methods in tune(): {bad_methods}")

    combine_values = ["mean"] if combine is None else [str(x).lower() for x in _n._grid_list(combine)]
    if any(value not in {"mean", "median", "max"} for value in combine_values):
        raise ValueError("combine must contain only 'mean', 'median', and/or 'max'.")

    adjust = {key: list(vals) for key, vals in _n._DEFAULT_TUNE_ADJUST_GRID.items()}
    for key, value in {
        "auto_target": auto_target,
        "thr_norm": thr_norm,
        "thr_raw": thr_raw,
        "k": k,
        "mutual": mutual,
        "attach_isolates": attach_isolates,
        "min_layers": min_layers,
    }.items():
        if value is not None:
            adjust[key] = _n._grid_list(value)

    search = {
        "layer_modes": layer_mode_values,
        "graphs": graph_values,
        "methods": method_values,
        "method_grids": method_grid_map,
        "combine": list(dict.fromkeys(combine_values)),
        "auto_targets": [float(v) for v in adjust["auto_target"] if v is not None],
        "thr_norm": [float(v) for v in adjust["thr_norm"] if v is not None],
        "thr_raw": [float(v) for v in adjust["thr_raw"] if v is not None],
        "k": list(dict.fromkeys(adjust["k"])),
        "mutual": [bool(v) for v in adjust["mutual"]],
        "attach_isolates": [bool(v) for v in adjust["attach_isolates"]],
        "min_layers": [int(v) for v in adjust["min_layers"] if v is not None],
        "refine_mutual": [bool(v) for v in adjust["mutual"]] if mutual is not None else [False, True],
    }
    configs: List[Dict[str, Any]] = []
    build_id = 0
    for layer_mode in search["layer_modes"]:
        targets = [g for g in search["graphs"] if g in _n._VALID_TUNE_GRAPHS[node_mode][layer_mode]]
        if not targets:
            continue
        combines = search["combine"] if (node_mode == "samples" and layer_mode == "multilayer") else [None]
        for method in search["methods"]:
            grid = {k: _n._grid_list(v) for k, v in search["method_grids"].get(method, {}).items()}
            keys = list(grid)
            rows = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])] if keys else [{}]
            for infer_kwargs in rows:
                for combine in combines:
                    build_id += 1
                    build_params = {
                        "method": method,
                        "node_mode": node_mode,
                        "layer_mode": layer_mode,
                        "graph": "entire",
                        "community_res": float(community_res),
                        "n_jobs": int(n_jobs),
                        **infer_kwargs,
                    }
                    if combine is not None:
                        build_params["combine"] = combine
                    configs.append(
                        {
                            "build_id": build_id,
                            "method": method,
                            "layer_mode": layer_mode,
                            "combine": combine,
                            "infer_kwargs": infer_kwargs,
                            "targets": list(targets),
                            "build_params": build_params,
                        }
                    )
    return search, configs


def _refine_window_impl(center: Optional[float], *, mode: str = "norm", low: float = 0.0, high: float = 1.0) -> List[float]:
    if center is None or not np.isfinite(center):
        return []
    if mode == "raw":
        vals = [center * 0.93, center * 0.97, center, center * 1.03, center * 1.07]
    elif mode == "auto":
        vals = [center - 0.02, center - 0.01, center, center + 0.01, center + 0.02]
    else:
        vals = [center - 0.02, center - 0.01, center, center + 0.01, center + 0.02]
    return sorted({round(float(np.clip(v, low, high)), 6) for v in vals})


def _tune_adjust_candidates_impl(
    target_graph: str,
    search: Dict[str, Any],
    *,
    seed: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    entries: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()

    def push(family: str, params: Dict[str, Any]) -> None:
        clean = dict(params)
        if target_graph != "consensus":
            clean.pop("min_layers", None)
        key = (family, tuple(sorted(clean.items(), key=lambda x: x[0])))
        if key not in seen:
            seen.add(key)
            entries.append((family, clean))

    if seed is None:
        attach_opts = search["attach_isolates"]
        k_opts = search["k"]
        min_opts = search["min_layers"] if (target_graph == "consensus" and search["min_layers"]) else [None]
        families = [("auto", {"auto_target": float(v)}) for v in search["auto_targets"]]
        raw_opts = [None, *search["thr_raw"]] if search["thr_raw"] else [None]
        norm_opts = [None, *search["thr_norm"]] if search["thr_norm"] else [None]
        families.extend(
            [
                ("manual", {"thr_raw": thr_raw_val, "thr_norm": thr_norm_val})
                for thr_raw_val, thr_norm_val in product(raw_opts, norm_opts)
                if not (thr_raw_val is None and thr_norm_val is None)
            ]
        )
        mutual_base = search["mutual"]
    else:
        attach_opts = [bool(seed.get("attach_isolates", False))]
        k_opts = [seed.get("k_input"), None, "auto"]
        if seed.get("_k_effective") is not None and pd.notna(seed.get("_k_effective")):
            k_now = int(seed["_k_effective"])
            k_opts.extend([max(2, k_now - 2), max(2, k_now - 1), k_now, min(10, k_now + 1), min(10, k_now + 2)])
        k_opts = list(dict.fromkeys(k_opts))
        min_opts = [int(seed["min_layers"])] if (target_graph == "consensus" and seed.get("min_layers") is not None) else [None]
        families = (
            [("auto", {"auto_target": v}) for v in (_refine_window_impl(float(seed.get("auto_target", 0.95)), mode="auto", low=1e-6, high=1.0) or [float(seed.get("auto_target", 0.95))])]
            if seed.get("family") == "auto"
            else [
                ("manual", {"thr_raw": thr_raw_val, "thr_norm": thr_norm_val})
                for thr_raw_val, thr_norm_val in product(
                    ([None] if seed.get("thr_raw_input") is None else _refine_window_impl(seed.get("thr_raw_input"), mode="raw")),
                    ([None] if seed.get("thr_norm_input") is None else _refine_window_impl(float(seed["thr_norm_input"]), mode="norm", low=0.0, high=1.0)),
                )
                if not (thr_raw_val is None and thr_norm_val is None)
            ]
        )
        mutual_base = search["refine_mutual"]

    for family, base in families:
        for kval, attach_val in product(k_opts, attach_opts):
            mutual_opts = mutual_base if kval is not None else [False]
            for mutual_val in mutual_opts:
                common = {"k": kval, "mutual": bool(mutual_val), "attach_isolates": bool(attach_val)}
                for min_val in min_opts:
                    extra = {} if min_val is None else {"min_layers": int(min_val)}
                    push(family, {**base, **common, **extra})
    return entries or [("base", {})]


def _clone_tune_candidate_impl(base_nt: "Any") -> "Any":
    cand = base_nt.__class__(
        rodins=base_nt.rodins,
        names=base_nt.names,
        samples=base_nt.sample_ids,
        fig=None,
        _meta=base_nt._meta,
        _cache=base_nt._cache,
    )
    cand._state = dict(base_nt._state)
    for name, entry in base_nt._graph_store.items():
        graph_obj = entry.get("graph")
        cand._set_graph_entry(
            name,
            graph=None if graph_obj is None else graph_obj.copy(),
            info=entry.get("info"),
            stats=entry.get("stats"),
            edges=entry.get("edges"),
        )
    return cand


def _evaluate_tune_candidate_impl(
    base_nt: "Any",
    cfg: Dict[str, Any],
    graph_name: str,
    family: str,
    adjust_params: Dict[str, Any],
    *,
    objective: Optional[str] = None,
    label_map: Optional[Dict[str, Any]] = None,
    include_metrics: bool = False,
) -> Dict[str, Any]:
    cand = _clone_tune_candidate_impl(base_nt)
    if adjust_params:
        _silent(cand.adjust, graph=graph_name, **adjust_params)
    else:
        cand.set_graph(graph_name)
    metrics = _graph_metrics_impl(cand, graph_name, objective=str(objective), label_map=label_map) if include_metrics else None
    info = cand._graph_info(graph_name)
    semantics = _graph_semantics(
        {
            "nodeMode": cfg["build_params"].get("node_mode"),
            "layerMode": cfg["layer_mode"],
        },
        graph_name,
    )
    row = {
        "build_id": cfg["build_id"],
        "layer_mode": cfg["layer_mode"],
        "graph": graph_name,
        "method": cfg["method"],
        "method_params": _fmt_method_params(cfg["infer_kwargs"]),
        "family": family,
        "auto_target": info.get("auto_target"),
        "auto": (info.get("auto_info") or {}).get("strategy"),
        "thr_raw": info.get("thr_raw"),
        "thr_norm": info.get("thr_norm"),
        "thr_raw_input": info.get("thr_raw_input"),
        "thr_norm_input": info.get("thr_norm_input"),
        "k_input": _requested_k(info),
        "k": info.get("k"),
        "mutual": bool(info.get("mutual", False)),
        "attach_isolates": bool(info.get("attach_isolates", False)),
        "combine": cfg["combine"],
        "min_layers": info.get("min_layers"),
        "_kind": semantics["kind"],
        "_k_effective": info.get("k"),
        **_public_param_state(info, semantics, nt=cand, graph_name=graph_name),
        "_build_params": dict(cfg["build_params"]),
        "_adjust_params": dict(adjust_params),
    }
    if include_metrics:
        row.update({key: val for key, val in metrics.items() if not str(key).startswith("_")})
        row["_edge_set"] = metrics["_edge_set"]
        row["_module_map"] = metrics["_module_map"]
    else:
        stats = cand._graph_stats(graph_name)
        num_nodes = int(stats.get("numNodes", 0))
        isolate_count = int(info.get("isolate_count", max(num_nodes - int(stats.get("nodesWithEdges", 0)), 0)))
        row["isolate_fraction"] = float(isolate_count / num_nodes) if num_nodes else 0.0
    return _mask_inactive_threshold_fields(row)


def _update_tune_stability_impl(
    rows: List[Dict[str, Any]],
    *,
    groups: Optional[Sequence[Sequence[Any]]] = None,
) -> None:
    if groups is None:
        grouped_rows: Iterable[Sequence[Dict[str, Any]]] = (
            group_rows
            for _, group_rows in (
                (group_name, [row for row in rows if str(row.get("_group")) == group_name])
                for group_name in dict.fromkeys(str(row.get("_group")) for row in rows)
            )
        )
    else:
        by_key = {row.get("_candidate_key"): row for row in rows}
        peer_keys: Dict[Any, set] = {key: set() for key in by_key}
        for group in groups:
            members = [key for key in dict.fromkeys(group) if key in by_key]
            if len(members) <= 1:
                continue
            for key in members:
                peer_keys[key].update(other for other in members if other != key)
        grouped_rows = (
            [by_key[key], *[by_key[peer] for peer in sorted(peers, key=str)]]
            for key, peers in peer_keys.items()
            if peers
        )
    for group_rows in grouped_rows:
        row = group_rows[0]
        others = list(group_rows[1:])
        if not others:
            continue
        row["edge_stability"] = float(np.mean([
            len(row["_edge_set"] & other["_edge_set"]) / len(row["_edge_set"] | other["_edge_set"])
            if (row["_edge_set"] | other["_edge_set"]) else 1.0
            for other in others
        ]))
        module_scores = []
        for other in others:
            shared = sorted(set(row["_module_map"]) & set(other["_module_map"]))
            if len(shared) < 2:
                module_scores.append(0.0)
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                module_scores.append(
                    float(
                        normalized_mutual_info_score(
                            [row["_module_map"][node] for node in shared],
                            [other["_module_map"][node] for node in shared],
                        )
                    )
                )
        row["module_stability"] = float(np.mean(module_scores))


def _prepare_grid_rows_for_scoring_impl(
    nt: "Any",
    grid: Dict[str, Any],
    *,
    objective: str,
    label_name: Optional[str],
) -> List[Dict[str, Any]]:
    def hydrate(rows_in: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for src in rows_in:
            row = dict(src)
            row.pop("module_purity", None)
            if ("module_size_band" not in row) or (not np.isfinite(float(row.get("module_size_band", np.nan)))):
                row["module_size_band"] = _module_size_band_score_impl(
                    int(row.get("active_nodes", 0)),
                    int(row.get("modules", 0)),
                )
            out_rows.append(row)
        return out_rows

    cache = grid.setdefault("score_cache", {})
    cache_key = (str(objective), str(label_name) if label_name is not None else None)
    cached = cache.get(cache_key)
    if cached is not None:
        hydrated = hydrate(cached)
        cache[cache_key] = [dict(row) for row in hydrated]
        return hydrated
    rows = hydrate([dict(row) for row in grid.get("rows", [])])
    if objective == "supervised":
        if label_name is None:
            raise ValueError("label must be provided for supervised grid scoring.")
        label_map = _sample_label_map_impl(nt, str(label_name))
        for row in rows:
            row.update(
                _supervised_terms_impl(
                    module_map=row.get("_module_map") or {},
                    edge_set=row.get("_edge_set"),
                    label_map=label_map,
                )
            )
    cache[cache_key] = [dict(row) for row in rows]
    return rows


def _tune_row_signature_impl(row: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    def norm(value: Any) -> Any:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    keys = (
        "graph", "family", "combine", "min_layers", "thr_raw", "thr_norm", "thr_raw_input",
        "thr_norm_input", "auto_target", "auto", "attach_isolates", "k_input", "k", "mutual",
        "thr_raw_base", "thr_norm_base", "auto_base", "k_input_base", "k_base", "attach_isolates_base",
    )
    return tuple((key, norm(row.get(key))) for key in keys if key in row)


def _score_snapshot_row_impl(
    nt: "Any",
    graph_name: str,
    *,
    objective: str,
    label_map: Optional[Dict[str, Any]],
    group: str,
) -> Dict[str, Any]:
    info = nt._graph_info(graph_name)
    semantics = _graph_semantics(nt._meta or {}, graph_name)
    metrics = _graph_metrics_impl(nt, graph_name, objective=objective, label_map=label_map)
    row = {
        "graph": graph_name,
        "family": ("auto" if (info.get("auto_info") or {}).get("strategy") else "manual"),
        "auto_target": info.get("auto_target"),
        "auto": (info.get("auto_info") or {}).get("strategy"),
        "thr_raw": info.get("thr_raw"),
        "thr_norm": info.get("thr_norm"),
        "thr_raw_input": info.get("thr_raw_input"),
        "thr_norm_input": info.get("thr_norm_input"),
        "k_input": _requested_k(info),
        "k": info.get("k"),
        "mutual": bool(info.get("mutual", False)),
        "attach_isolates": bool(info.get("attach_isolates", False)),
        "combine": info.get("combine"),
        "min_layers": info.get("min_layers"),
        "_kind": semantics.get("kind", "direct"),
        "_k_effective": info.get("k"),
        **_public_param_state(info, semantics, nt=nt, graph_name=graph_name),
        **{key: val for key, val in metrics.items() if not str(key).startswith("_")},
        "_group": group,
        "_edge_set": metrics["_edge_set"],
        "_module_map": metrics["_module_map"],
    }
    return _mask_inactive_threshold_fields(row)


def _build_tune_grid_impl(
    nt: "Any",
    *,
    node_mode: str = "samples",
    layer_modes: Optional[Union[str, Sequence[str]]] = None,
    graphs: Optional[Union[str, Sequence[str]]] = None,
    methods: Optional[Sequence[str]] = None,
    method_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
    combine: Optional[Union[str, Sequence[str]]] = None,
    auto_target: Optional[Union[float, Sequence[float]]] = None,
    thr_norm: Optional[Union[float, Sequence[float]]] = None,
    thr_raw: Optional[Union[float, Sequence[float]]] = None,
    k: Optional[Union[Any, Sequence[Any]]] = None,
    mutual: Optional[Union[bool, Sequence[bool]]] = None,
    attach_isolates: Optional[Union[bool, Sequence[bool]]] = None,
    min_layers: Optional[Union[int, Sequence[int]]] = None,
    community_res: float = 1.0,
    verbose: bool = True,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    node_mode = str(node_mode).lower()
    if node_mode not in {"samples", "features"}:
        raise ValueError("node_mode must be 'samples' or 'features'.")

    search, build_configs = _prepare_tune_configs_impl(
        node_mode=node_mode,
        n_layers=len(nt.rodins),
        layer_modes=layer_modes,
        graphs=graphs,
        methods=methods,
        method_grids=method_grids,
        combine=combine,
        auto_target=auto_target,
        thr_norm=thr_norm,
        thr_raw=thr_raw,
        k=k,
        mutual=mutual,
        attach_isolates=attach_isolates,
        min_layers=min_layers,
        community_res=float(community_res),
        n_jobs=int(n_jobs),
    )
    if not build_configs:
        raise ValueError("No valid build candidates were produced for the requested tune search.")

    failures: List[Dict[str, Any]] = []
    base_by_build: Dict[int, Any] = {}
    build_by_id = {cfg["build_id"]: cfg for cfg in build_configs}
    coarse_rows: List[Dict[str, Any]] = []

    if verbose:
        print("[Netan.grid] " + " | ".join(["stage=1/2", f"node_mode={node_mode}", f"builds={len(build_configs)}"]))
    for idx, cfg in enumerate(build_configs, start=1):
        if verbose:
            parts = ["stage=1/2", f"iter={idx}/{len(build_configs)}", f"method={cfg['method']}", f"layer_mode={cfg['layer_mode']}", f"graphs={','.join(cfg['targets'])}"]
            if cfg["infer_kwargs"]:
                parts.append(f"method_params={_fmt_method_params(cfg['infer_kwargs'])}")
            if cfg["combine"] is not None:
                parts.append(f"combine={cfg['combine']}")
            print("[Netan.grid] " + " | ".join(parts))
        base_nt = nt.__class__(rodins=list(nt.rodins), names=list(nt.names), samples=list(nt.sample_ids))
        try:
            _silent(base_nt.build, **cfg["build_params"])
        except Exception as exc:
            failures.append({"stage": "build", "build_id": cfg["build_id"], "method": cfg["method"], "layer_mode": cfg["layer_mode"], "graph": None, "error": str(exc)})
            continue
        base_by_build[cfg["build_id"]] = base_nt
        for graph_name in cfg["targets"]:
            for family, adjust_params in _tune_adjust_candidates_impl(graph_name, search):
                try:
                    coarse_rows.append(_evaluate_tune_candidate_impl(base_nt, cfg, graph_name, family, adjust_params))
                except Exception as exc:
                    failures.append({"stage": "coarse", "build_id": cfg["build_id"], "method": cfg["method"], "layer_mode": cfg["layer_mode"], "graph": graph_name, "error": str(exc)})
    if not coarse_rows:
        if failures:
            raise RuntimeError(f"All grid candidates failed. First error: {failures[0]['error']}")
        raise RuntimeError("No grid candidates were evaluated successfully.")
    if verbose:
        print("[Netan.grid] " + " | ".join(["stage=1/2 done", f"ok_builds={len(base_by_build)}", f"coarse={len(coarse_rows)}", f"failures={len(failures)}"]))

    final_rows: List[Dict[str, Any]] = []
    seen_final_exact = set()
    seen_final_states = set()
    exact_to_state: Dict[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]] = {}
    if verbose:
        print("")
        print("[Netan.grid] " + " | ".join(["stage=2/2", f"refine_seeds={len(coarse_rows)}"]))
    stage2_bar = tqdm(total=len(coarse_rows), desc="Grid 2/2", leave=False) if verbose else None
    refine_groups: List[Tuple[Any, ...]] = []
    candidate_id = 0
    for seed in coarse_rows:
        cfg = build_by_id[seed["build_id"]]
        refine_candidates = _tune_adjust_candidates_impl(str(seed["graph"]), search, seed=seed)
        if stage2_bar is not None:
            suffix = [f"{cfg['method']}:{cfg['layer_mode']}:{seed['graph']}", f"family={seed.get('family', '-')}", f"adjust={len(refine_candidates)}"]
            if cfg["infer_kwargs"]:
                suffix.append(_fmt_method_params(cfg["infer_kwargs"]))
            stage2_bar.set_postfix_str(" | ".join(suffix))
        group_keys: List[Tuple[Any, ...]] = []
        for family, adjust_params in refine_candidates:
            try:
                row = _evaluate_tune_candidate_impl(
                    base_by_build[seed["build_id"]],
                    cfg,
                    str(seed["graph"]),
                    family,
                    adjust_params,
                    objective="unsupervised",
                    label_map=None,
                    include_metrics=True,
                )
            except Exception as exc:
                failures.append({"stage": "refine", "build_id": cfg["build_id"], "method": cfg["method"], "layer_mode": cfg["layer_mode"], "graph": str(seed["graph"]), "error": str(exc)})
                continue
            exact_sig = _tune_row_signature_impl(row)
            edge_set = row.get("_edge_set")
            state_sig = tuple(
                (key, tuple(sorted(edge_set)) if key == "_edge_set" and isinstance(edge_set, set) else row.get(key))
                for key in ("graph", "_edge_set")
                if key in row
            )
            if exact_sig in seen_final_exact or state_sig in seen_final_states:
                if exact_sig not in exact_to_state and state_sig in seen_final_states:
                    group_keys.append(next((k for k, v in exact_to_state.items() if v == state_sig), exact_sig))
                continue
            seen_final_exact.add(exact_sig)
            seen_final_states.add(state_sig)
            exact_to_state[exact_sig] = state_sig
            row["_candidate_key"] = exact_sig
            candidate_id += 1
            row["candidate_id"] = int(candidate_id)
            group_keys.append(exact_sig)
            final_rows.append(row)
        if len(group_keys) > 1:
            refine_groups.append(tuple(group_keys))
        if stage2_bar is not None:
            stage2_bar.update(1)
    if stage2_bar is not None:
        stage2_bar.close()
    if not final_rows:
        raise RuntimeError("All refined grid candidates failed.")

    _update_tune_stability_impl(final_rows, groups=refine_groups)
    public_table = pd.DataFrame([{k: v for k, v in row.items() if not str(k).startswith("_")} for row in final_rows])
    fail_df = pd.DataFrame(failures)
    grid = {
        "kind": "tune_grid",
        "node_mode": node_mode,
        "search": search,
        "build_configs": build_configs,
        "build_by_id": build_by_id,
        "base_by_build": base_by_build,
        "rows": final_rows,
        "coarse_candidates": int(len(coarse_rows)),
        "final_candidates": int(len(final_rows)),
        "refine_groups": refine_groups,
        "failures": failures,
        "failures_table": fail_df,
        "table": public_table,
        "score_cache": {},
    }
    nt._cache["tune_grid"] = grid
    nt._results()["grid"] = {
        "node_mode": node_mode,
        "coarse_candidates": int(len(coarse_rows)),
        "final_candidates": int(len(final_rows)),
        "num_failures": int(len(failures)),
        "table": _n._trim_public_table(public_table.copy()),
        "failures": fail_df,
    }
    if verbose:
        print("[Netan.grid] " + " | ".join(["done", f"coarse={len(coarse_rows)}", f"final={len(final_rows)}", f"failures={len(failures)}"]))
    return grid


def _score_current_graph_impl(
    nt: "Any",
    graph_name: str,
    *,
    objective: str,
    label_map: Optional[Dict[str, Any]],
    weights: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_row = _score_snapshot_row_impl(nt, graph_name, objective=objective, label_map=label_map, group="scores")
    rows = [base_row]
    seen = {_tune_row_signature_impl(base_row)}
    for family, adjust_params in _tune_adjust_candidates_impl(graph_name, {"refine_mutual": [False, True]}, seed=base_row):
        cand = _clone_tune_candidate_impl(nt)
        if adjust_params:
            _silent(cand.adjust, graph=graph_name, **adjust_params)
        else:
            cand.set_graph(graph_name)
        row = _score_snapshot_row_impl(cand, graph_name, objective=objective, label_map=label_map, group="scores")
        row["family"] = family
        sig = _tune_row_signature_impl(row)
        if sig in seen:
            continue
        seen.add(sig)
        rows.append(row)
    _update_tune_stability_impl(rows)
    for row in rows:
        row.update(_score_terms_impl(row, objective, weights=weights))
    current_sig = _tune_row_signature_impl(base_row)
    return next((row for row in rows if _tune_row_signature_impl(row) == current_sig), rows[0])


def _score_tune_grid_impl(
    nt: "Any",
    grid: Dict[str, Any],
    *,
    label: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    top_results: int = 10,
    apply: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    if not isinstance(grid, dict) or str(grid.get("kind")) != "tune_grid":
        raise TypeError("grid must be a bundle returned by Netan.grid().")
    node_mode = str(grid.get("node_mode", "samples"))
    if node_mode == "features" and label is not None:
        raise ValueError("label is available only when node_mode='samples'.")
    objective_eff = "supervised" if label is not None else "unsupervised"
    label_name = str(label) if label is not None else None
    weights_public = {
        block: {name: round(float(val) * 100.0, 4) for name, val in vals.items()}
        for block, vals in _resolve_score_weights_impl(weights).items()
    }

    if verbose:
        print("[Netan.scores_grid] " + " | ".join([f"objective={objective_eff}", f"node_mode={node_mode}", f"candidates={int(grid.get('final_candidates', 0))}"]))

    final_rows = _prepare_grid_rows_for_scoring_impl(nt, grid, objective=objective_eff, label_name=label_name)
    for row in final_rows:
        row.update(_score_terms_impl(row, objective_eff, weights=weights))
    final_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    best_row = final_rows[0]

    out = pd.DataFrame([{k: v for k, v in row.items() if not str(k).startswith("_")} for row in final_rows])
    keep = min(max(int(top_results), 1), len(out))
    leaderboard = out.head(keep).reset_index(drop=True)
    fail_df = pd.DataFrame(grid.get("failures") or [])

    applied = False
    if apply:
        candidate_nt = _materialize_grid_candidate_impl(grid, candidate=best_row)
        _apply_materialized_candidate_impl(nt, candidate_nt)
        applied = True

    result = {
        "objective": objective_eff,
        "label": label_name,
        "score": float(best_row["score"]),
        "best_method": str(best_row["method"]),
        "best_layer_mode": str(best_row["layer_mode"]),
        "best_graph": str(best_row["graph"]),
        "best_candidate_id": int(best_row.get("candidate_id", 0)),
        "coarse_candidates": int(grid.get("coarse_candidates", 0)),
        "final_candidates": int(len(final_rows)),
        "num_failures": int(len(grid.get("failures") or [])),
        "applied": bool(applied),
        "best_build_params": dict(best_row["_build_params"]),
        "best_adjust_params": dict(best_row["_adjust_params"]),
        "best_metrics": {k: v for k, v in best_row.items() if not str(k).startswith("_")},
        "table": leaderboard,
        "failures": fail_df,
        "weights": weights_public,
    }
    nt._results()["scores_grid"] = result
    nt._results()["tune"] = result
    grid["last_scores"] = {
        "objective": objective_eff,
        "label": label_name,
        "leaderboard": leaderboard,
        "best_candidate_id": int(best_row.get("candidate_id", 0)),
        "weights": weights_public,
    }
    if verbose:
        print("[Netan.scores_grid] " + " | ".join(["done", f"best={best_row['method']}:{best_row['layer_mode']}:{best_row['graph']}", f"score={float(best_row['score']):.4f}", f"applied={bool(applied)}"]))
    return leaderboard


def _cached_tune_grid_impl(nt: "Any") -> Dict[str, Any]:
    grid = (getattr(nt, "_cache", {}) or {}).get("tune_grid")
    if not isinstance(grid, dict) or str(grid.get("kind")) != "tune_grid":
        raise RuntimeError("Run grid() first or pass an explicit grid bundle.")
    return grid


def _resolve_grid_candidate_impl(grid: Dict[str, Any], candidate: Optional[Any] = None) -> Dict[str, Any]:
    rows = list(grid.get("rows") or [])
    if not rows:
        raise RuntimeError("grid does not contain any candidates.")
    if candidate is None:
        last_scores = grid.get("last_scores") or {}
        best_candidate_id = last_scores.get("best_candidate_id")
        if best_candidate_id is None:
            raise RuntimeError("Run scores_grid() first or pass an explicit candidate to materialize().")
        match = next((row for row in rows if int(row.get("candidate_id", -1)) == int(best_candidate_id)), None)
        if match is not None:
            return match
        raise RuntimeError("The latest scores_grid() winner was not found in the provided grid.")
    if isinstance(candidate, pd.Series):
        candidate = candidate.to_dict()
    if isinstance(candidate, dict):
        if "candidate_id" in candidate:
            cand_id = int(candidate["candidate_id"])
            match = next((row for row in rows if int(row.get("candidate_id", -1)) == cand_id), None)
            if match is not None:
                return match
            raise ValueError(f"candidate_id '{cand_id}' was not found inside the provided grid.")
        else:
            signature = _tune_row_signature_impl(candidate)
            match = next((row for row in rows if _tune_row_signature_impl(row) == signature), None)
            if match is None:
                raise ValueError("candidate row was not found inside the provided grid.")
            return match
    if isinstance(candidate, (int, np.integer)):
        selector = int(candidate)
        last_scores = grid.get("last_scores") or {}
        leaderboard = last_scores.get("leaderboard")
        if isinstance(leaderboard, pd.DataFrame) and 0 <= selector < len(leaderboard):
            lead_row = leaderboard.iloc[selector]
            lead_candidate_id = lead_row.get("candidate_id")
            if pd.notna(lead_candidate_id):
                match = next((row for row in rows if int(row.get("candidate_id", -1)) == int(lead_candidate_id)), None)
                if match is not None:
                    return match
        match = next((row for row in rows if int(row.get("candidate_id", -1)) == selector), None)
        if match is not None:
            return match
        if 0 <= selector < len(rows):
            return rows[selector]
        raise ValueError(f"candidate '{selector}' was not found inside the provided grid.")
    raise TypeError("candidate must be None, a leaderboard index or candidate_id integer, or a row-like dict/Series.")


def _materialize_grid_candidate_impl(grid: Dict[str, Any], candidate: Optional[Any] = None) -> "Any":
    if not isinstance(grid, dict) or str(grid.get("kind")) != "tune_grid":
        raise TypeError("grid must be a bundle returned by Netan.grid().")
    row = _resolve_grid_candidate_impl(grid, candidate=candidate)
    base_nt = grid["base_by_build"][row["build_id"]]
    cand = _clone_tune_candidate_impl(base_nt)
    adjust_params = dict(row.get("_adjust_params") or {})
    graph_name = str(row["graph"])
    if adjust_params:
        _silent(cand.adjust, graph=graph_name, **adjust_params)
    else:
        cand.set_graph(graph_name)
    return cand


def _apply_materialized_candidate_impl(target_nt: "Any", candidate_nt: "Any") -> "Any":
    prev_results = dict(getattr(target_nt, "_results_cache", {}) or {})
    prev_cache = dict(getattr(target_nt, "_cache", {}) or {})
    target_nt.fig = None
    target_nt._meta = dict(getattr(candidate_nt, "_meta", {}) or {})
    target_nt._state = dict(getattr(candidate_nt, "_state", {}) or {})
    target_nt._cache = dict(getattr(candidate_nt, "_cache", {}) or {})
    if "tune_grid" in prev_cache:
        target_nt._cache["tune_grid"] = prev_cache["tune_grid"]
    target_nt._graph_store = {}
    for name, entry in getattr(candidate_nt, "_graph_store", {}).items():
        graph_obj = entry.get("graph")
        target_nt._set_graph_entry(
            name,
            graph=None if graph_obj is None else graph_obj.copy(),
            info=entry.get("info"),
            stats=entry.get("stats"),
            edges=entry.get("edges"),
        )
    target_nt._results_cache = dict(prev_results)
    return target_nt
