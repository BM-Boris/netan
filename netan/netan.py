"""Public Netan API for building, scoring, ranking, tuning, and exporting networks.

Internal build helpers live in ``_build.py``, internal view/sync helpers
live in ``_views.py``, ranking helpers live in ``_ranking.py``, and
scoring/tuning helpers live in ``_tuning.py``. This module keeps the
public container class and high-level orchestration layer.
"""

import copy
import pickle
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import networkx as nx
import numpy as np
import pandas as pd
import os
if ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_GPU" in os.environ):
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except Exception:
        pass

_UNSET = object()
_CORE_GRAPHS = ("entire", "fused", "consensus", "cross")
_FEATURE_SYNC_COLUMNS = (
    "rank",
    "top_class",
    "score",
    "p_perm",
    "p_adj",
    "stability_rank",
    "stability_top_class",
    "selected_freq",
    "mean_rank",
    "median_rank",
    "stability_score",
    "stability_score_sd",
)

class GraphEntry(TypedDict):
    graph: Optional[nx.Graph]
    info: Dict[str, Any]
    stats: Dict[str, Any]
    edges: Optional[pd.DataFrame]

class BuildState(TypedDict):
    graphs: Dict[str, nx.Graph]
    infos: Dict[str, Dict[str, Any]]
    matrices: Dict[str, Dict[str, np.ndarray]]
    tag_map: Dict[str, str]

from ._build import (
    BUILDERS,
    _annotate_feature_layers,
    _annotate_support_layers,
    _build_outputs,
    common_samples,
    _ensure_df,
    _extract_sample_ids_from_df,
    _feature_entire_graph,
    _feature_meta_lookup,
    _graph_edge_table,
    _graph_to_adjacency,
    _matrix_to_graph,
    _refresh_graph_entries,
    _requested_k,
    _samples_multilayer_derived,
    _set_single_layer_attrs,
    _sparsify_matrix,
    _subset_rodin,
    _support_base_info,
    _validate_layer_names,
)

def _adjust_sparsify_kwargs(
    base_info: Dict[str, Any],
    *,
    thr_raw=_UNSET,
    thr_norm=_UNSET,
    auto_target=_UNSET,
    attach_isolates=_UNSET,
    k=_UNSET,
    mutual=_UNSET,
) -> Dict[str, Any]:
    threshold_override = thr_raw is not _UNSET or thr_norm is not _UNSET
    enable_auto = auto_target is not _UNSET and not threshold_override
    if enable_auto:
        eff_thr_raw = None
        eff_thr_norm = None
    elif threshold_override:
        eff_thr_raw = None if thr_raw is _UNSET else thr_raw
        eff_thr_norm = None if thr_norm is _UNSET else thr_norm
    else:
        eff_thr_raw = base_info.get("thr_raw_input")
        eff_thr_norm = base_info.get("thr_norm_input")
    return {
        "thr_raw": eff_thr_raw,
        "thr_norm": eff_thr_norm,
        "auto_target": (
            base_info.get("auto_target", 0.95) if auto_target is _UNSET else auto_target
        ),
        "attach_isolates": (
            bool(base_info.get("attach_isolates", False))
            if attach_isolates is _UNSET
            else bool(attach_isolates)
        ),
        "k": _requested_k(base_info) if k is _UNSET else k,
        "mutual": bool(base_info.get("mutual", False)) if mutual is _UNSET else bool(mutual),
    }


def _rebuild_direct_graph(
    ids: Sequence[str],
    mats: Dict[str, np.ndarray],
    base_info: Dict[str, Any],
    *,
    use_abs: bool = False,
    thr_raw=_UNSET,
    thr_norm=_UNSET,
    auto_target=_UNSET,
    attach_isolates=_UNSET,
    k=_UNSET,
    mutual=_UNSET,
) -> Tuple[nx.Graph, np.ndarray, Dict[str, Any]]:
    sparsify_kwargs = _adjust_sparsify_kwargs(
        base_info,
        thr_raw=thr_raw,
        thr_norm=thr_norm,
        auto_target=auto_target,
        attach_isolates=attach_isolates,
        k=k,
        mutual=mutual,
    )
    W_for_threshold = mats["W_abs"] if use_abs else mats["W_raw"]
    adj, info = _sparsify_matrix(W_for_threshold, mats["W_norm"], **sparsify_kwargs)
    return _matrix_to_graph(ids, W_for_threshold, adj), adj, info


class Netan:
    """Container for inputs, stored graphs, and current graph selection."""
    def __init__(
        self,
        rodins: Sequence[object],
        names: Sequence[str],
        samples: Sequence[str],
        G: Optional[nx.Graph] = None,
        G_layers: Optional[Dict[str, nx.Graph]] = None,
        G_all: Optional[nx.Graph] = None,
        G_fused: Optional[nx.Graph] = None,
        G_con: Optional[nx.Graph] = None,
        G_cross: Optional[nx.Graph] = None,
        fig: Optional["FigureWidget"] = None,
        _meta: Optional[Dict] = None,
        _cache: Optional[Dict] = None,
    ):
        self.rodins = list(rodins)
        self.names = list(map(str, names))
        self.sample_ids = list(map(str, samples))
        self.fig = fig
        meta = {} if _meta is None else dict(_meta)
        active_graph = str(meta.pop("activeGraph", meta.pop("graph", "entire")) or "entire")
        meta.pop("availableGraphs", None)
        self._meta = meta
        self._state = {"activeGraph": active_graph}
        cache = {} if _cache is None else dict(_cache)
        self._results_cache = dict(cache.pop("results", {}) or {})
        self._cache = cache
        self._graph_store: Dict[str, GraphEntry] = {}
        for name, graph_obj in {
            "entire": G_all or G,
            "fused": G_fused,
            "consensus": G_con,
            "cross": G_cross,
        }.items():
            if graph_obj is not None:
                self._set_graph_entry(name, graph=graph_obj)
        for name, graph_obj in (G_layers or {}).items():
            if graph_obj is not None:
                self._set_graph_entry(str(name), graph=graph_obj)

    def __repr__(self) -> str:
        def _feature_rows(rodin: object) -> int:
            X = getattr(rodin, "X", None)
            shape = getattr(X, "shape", None)
            return int(shape[0]) if shape is not None and len(shape) >= 1 else 0

        def _active_feature_count(active_graph: str, node_mode: str) -> int:
            if node_mode == "features" and self.G is not None:
                return int(self.G.number_of_nodes())
            layer_counts = {str(name).lower(): _feature_rows(rodin) for name, rodin in zip(self.names, self.rodins)}
            return layer_counts.get(str(active_graph).lower(), sum(layer_counts.values()))

        lines = ["< Netan object >"]
        lines.append("rodins: " + (", ".join(map(str, self.names)) if self.names else "-"))
        if self.G is None:
            total_features = sum(_feature_rows(rodin) for rodin in self.rodins)
            lines.append(f"samples: {len(self.sample_ids)} | features: {total_features}")
            lines.append("state: not built")
            return "\n".join(lines)

        active = self._active_graph_name()
        graphs = ", ".join(self.available_graphs(detailed=False))
        node_mode = str((self._meta or {}).get("nodeMode", "samples"))
        lines.append(f"samples: {len(self.sample_ids)} | features: {_active_feature_count(active, node_mode)}")
        lines.append(f"active_graph: {active}")
        lines.append(f"graphs: {graphs}")
        lines.append(f"node_mode: {node_mode}")
        lines.append(f"method: {(self._meta or {}).get('networkMethod', '?')}")
        G = self.G
        lines.append(f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        return "\n".join(lines)

    def copy(self) -> "Netan":
        """
        Return a deep copy of the current container.

        Returns
        -------
        Netan
            Independent copy of the current object. Inputs, graph variants,
            build metadata, and cached analysis results are copied. The
            interactive figure handle is cleared in the copy.
        """
        fig = self.fig
        self.fig = None
        try:
            out = copy.deepcopy(self)
        finally:
            self.fig = fig
        out.fig = None
        return out

    def _active_graph_name(self) -> str:
        return str((self._state or {}).get("activeGraph", "entire"))

    def _results(self) -> Dict[str, Any]:
        return self._results_cache

    @property
    def G(self) -> Optional[nx.Graph]:
        return self._graph_obj(self._active_graph_name())

    @property
    def G_all(self) -> Optional[nx.Graph]:
        return self._graph_obj("entire")

    @property
    def G_fused(self) -> Optional[nx.Graph]:
        return self._graph_obj("fused")

    @property
    def G_con(self) -> Optional[nx.Graph]:
        return self._graph_obj("consensus")

    @property
    def G_cross(self) -> Optional[nx.Graph]:
        return self._graph_obj("cross")

    @property
    def G_layers(self) -> Dict[str, nx.Graph]:
        return {
            name: entry["graph"]
            for name, entry in self._graph_store.items()
            if name not in _CORE_GRAPHS and entry.get("graph") is not None
        }

    def _graph_obj(self, name: str) -> Optional[nx.Graph]:
        return self._graph_store.get(str(name), {}).get("graph")

    def _graph_info(self, name: str) -> Dict[str, Any]:
        return dict(self._graph_store.get(str(name), {}).get("info") or {})

    def _graph_stats(self, name: str) -> Dict[str, Any]:
        return dict(self._graph_store.get(str(name), {}).get("stats") or {})

    def _set_graph_entry(
        self,
        name: str,
        *,
        graph: Optional[nx.Graph] = None,
        info: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        edges: Optional[pd.DataFrame] = None,
    ) -> None:
        key = str(name)
        if all(val is None for val in (graph, info, stats, edges)):
            self._graph_store.pop(key, None)
            return
        entry = self._graph_store.setdefault(key, {"graph": None, "info": {}, "stats": {}, "edges": None})
        if graph is not None:
            entry["graph"] = graph
            if edges is None:
                entry["edges"] = None
        if info is not None:
            entry["info"] = dict(info)
        if stats is not None:
            entry["stats"] = dict(stats)
        if edges is not None:
            entry["edges"] = edges.copy()

    def available_layers(self) -> List[str]:
        """
        Return edge-layer labels visible on the active graph.

        Returns
        -------
        list[str]
            Sorted layer labels present on active graph edges.
        """
        if self.G is None:
            return list(map(str, self.names))
        s = set()
        for _, _, d in self.G.edges(data=True):
            lays = d.get("layers") or {d.get("layer", "entire")}
            for L in lays:
                s.add(str(L))
        return sorted(s)

    def available_graphs(self, detailed: bool = True) -> Union[Dict[str, str], List[str]]:
        """
        Return the graph variants available in the current object.

        Parameters
        ----------
        detailed : bool, default=True
            If ``True``, return a mapping from public graph names to internal
            storage labels. If ``False``, return only selectable graph names.

        Returns
        -------
        dict or list
            Graph names, optionally with their internal storage labels.
        """
        graphs = {
            name: attr
            for name, attr in {"entire": "G_all", "fused": "G_fused", "consensus": "G_con", "cross": "G_cross"}.items()
            if self._graph_obj(name) is not None
        }
        for name, entry in self._graph_store.items():
            if name in _CORE_GRAPHS or entry.get("graph") is None:
                continue
            graphs[str(name)] = f"G_layers[{name!r}]"
        return graphs if detailed else list(graphs.keys())

    graphs = available_graphs

    def _resolve_active_graph(self, graph: Optional[str]) -> Tuple[str, nx.Graph]:
        if not self._graph_store:
            raise RuntimeError("Build the network first (.build).")

        choice = "entire" if graph in (None, "") else str(graph)
        lowered = choice.lower()
        key = {name: name for name in _CORE_GRAPHS}.get(lowered)
        if key is None:
            key = next(
                (name for name, entry in self._graph_store.items() if name.lower() == lowered and name not in _CORE_GRAPHS and entry.get("graph") is not None),
                None,
            )
        G = None if key is None else self._graph_obj(key)
        if G is None:
            raise ValueError(f"Unknown graph '{graph}'. Allowed graphs: {self.available_graphs(detailed=False)}")
        return key, G

    def _graph_context(self, graph: Optional[str]) -> Tuple[str, nx.Graph, Dict[str, Any]]:
        if not self._graph_store:
            raise RuntimeError("Build the network first (.build).")
        graph_name, G = self._resolve_active_graph(graph or self._active_graph_name())
        return graph_name, G, self._graph_info(graph_name)

    def set_graph(self, graph: Optional[str] = "entire") -> "Netan":
        """
        Switch the active graph used by plotting, tables, and export.

        Parameters
        ----------
        graph : str, default='entire'
            Graph variant name such as ``entire``, ``fused``, ``consensus``,
            ``cross``, or a layer name.

        Returns
        -------
        Netan
            The same object with a different active graph.
        """
        graph_name, _ = self._resolve_active_graph(graph)
        self.fig = None
        self._state["activeGraph"] = graph_name
        return self

    def _store_build_outputs(
        self,
        outputs: BuildState,
        *,
        method: str,
        thr_raw: Optional[float],
        thr_norm: Optional[float],
        auto_target: float,
        attach_isolates: bool,
        layer_mode: str,
        node_mode: str,
        combine: str,
        k: Optional[Union[int, str]],
        mutual: bool,
        min_layers: int,
        community_res: float,
    ) -> None:
        prev_results = dict(getattr(self, "_results_cache", {}) or {})
        all_graphs = dict(outputs["graphs"])
        for info in outputs["infos"].values():
            info["community_res"] = float(community_res)
        self._graph_store = {}
        _refresh_graph_entries(
            self,
            all_graphs,
            outputs["infos"],
            node_mode=node_mode,
            community_res=community_res,
        )

        self._meta = dict(
            networkMethod=method,
            thrRaw=(None if thr_raw is None else float(thr_raw)),
            thrNorm=(None if thr_norm is None else float(thr_norm)),
            autoTarget=float(auto_target),
            attachIsolates=bool(attach_isolates),
            layerMode=layer_mode,
            nodeMode=node_mode,
            combine=combine,
            kRequested=k,
            mutual=bool(mutual),
            minLayers=min_layers,
            communityResolution=float(community_res),
        )
        self._state = {"activeGraph": "entire"}
        self._cache = {
            **{key: self._cache[key] for key in ("feature_meta", "aligned_samples_df", "label_maps") if key in self._cache},
            "matrices": outputs["matrices"],
            "tag_map": outputs["tag_map"],
        }
        self._results_cache = {
            key: prev_results[key]
            for key in ("rank", "rank_stability", "shortlist")
            if key in prev_results
        }

    def build(
        self,
        *,
        method: str = "spearman",
        node_mode: str = "samples",
        layer_mode: str = "stack",
        thr_raw: Optional[float] = None,
        thr_norm: Optional[float] = None,
        auto_target: float = 0.95,
        attach_isolates: bool = False,
        k: Optional[Union[int, str]] = "auto",
        mutual: bool = False,
        min_layers: Optional[int] = None,
        combine: str = "mean",
        graph: str = "entire",
        community_res: float = 1.0,
        n_jobs: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> "Netan":
        """
        Build network graphs from the current inputs.

        Computes similarity matrices, applies thresholding and optional kNN
        sparsification, stores the graph variants required by the selected
        mode, and sets the active graph.

        Parameters
        ----------
        method : {'spearman', 'clr', 'rf', 'glasso'}, default='spearman'
            Similarity method.
        node_mode : {'samples', 'features'}, default='samples'
            Whether nodes represent samples or features.
        layer_mode : {'stack', 'multilayer'}, default='stack'
            Whether inputs are combined into one graph or kept as a multilayer
            build.
        thr_raw : float, default=None
            Threshold on ``abs(W_raw)``.
        thr_norm : float, default=None
            Threshold on normalized similarity ``W_norm``.
        auto_target : float, default=0.95
            Target active-node fraction when no manual threshold is supplied.
        attach_isolates : bool, default=False
            Reattach isolated nodes with one strongest edge when possible.
        k : int, 'auto', or None, default='auto'
            Optional kNN restriction after thresholding.
        mutual : bool, default=False
            Use mutual-kNN instead of union-kNN when k is active.
        min_layers : int, default=None
            Minimum support for multilayer sample consensus graphs.
        combine : {'mean', 'median', 'max'}, default='mean'
            Fusion rule for multilayer aggregation.
        graph : str, default='entire'
            Graph variant to select after the build.
        community_res : float, default=1.0
            Louvain resolution for module labels.
        n_jobs : int, default=1
            Parallelism for inference methods that support it.
        verbose : bool, default=True
            If ``True``, print the compact build summary.
        **kwargs
            Method-specific settings such as CLR neighbors, RF trees, or
            glasso settings.

        Notes
        -----
        Method-specific ``kwargs`` are passed only to the selected ``method``.
        Typical examples are ``method='clr', n_neighbors=5``,
        ``method='rf', n_estimators=320, max_depth=8``, and
        ``method='glasso', alpha=0.05, max_iter=200, tol=1e-4``.

        ``thr_raw`` is method-scale specific, so there is no universal cutoff
        across methods. Prefer ``thr_norm`` when you want more comparable
        behavior across ``spearman``, ``clr``, ``rf``, and ``glasso``.
        Rough starting ranges for ``thr_raw`` are often around ``0.3`` to
        ``0.8`` for ``spearman`` and ``0.01`` to ``0.2`` for ``glasso``.
        In samples mode, Netan also syncs graph-level ``community`` and
        ``module`` labels back into each ``rodin.samples`` table under
        ``netan_<graph>_...`` columns.

        Methodology
        -----------
        Each method first computes a full similarity matrix ``W_raw``. Netan
        then derives ``W_norm``, a monotonic normalization of ``abs(W_raw)``
        to ``[0, 1]``. If both ``thr_raw`` and ``thr_norm`` are provided, an
        edge must pass both. ``k`` is applied after thresholding.

        When ``k='auto'``, Netan inspects the thresholded graph: if
        ``mean_degree_active <= 5``, no extra kNN pruning is used; otherwise
        it sets ``k = 3 + floor(sqrt(mean_degree_active - 5))``, capped by
        ``ceil(median_degree_active)``, ``active_nodes - 1``, and ``10``.

        In multilayer samples mode, layer graphs are direct, ``fused`` is
        built from fused full layer matrices and then sparsified, and
        ``entire`` and ``consensus`` are derived from layer graphs. In
        multilayer features mode, layer graphs and ``cross`` are direct, while
        ``entire`` is derived from layers and ``cross``.

        Returns
        -------
        Netan
            The same object with graph variants and summaries updated.
        """
        method = method.lower()
        node_mode = node_mode.lower()
        layer_mode = layer_mode.lower()
        combine = combine.lower()
        graph_choice = "entire" if graph in (None, "") else str(graph).lower()

        if len(self.rodins) < 2 and layer_mode == "multilayer":
            raise ValueError("For multilayer mode -> multiple rodins should be provided.")
        if method not in BUILDERS:
            raise ValueError(f"Unknown method '{method}'. Allowed: {list(BUILDERS)}.")
        if node_mode not in ("samples", "features"):
            raise ValueError("node_mode must be 'samples' or 'features'.")
        if layer_mode not in ("stack", "multilayer"):
            raise ValueError("layer_mode must be 'stack' or 'multilayer'.")
        if layer_mode == "stack" and graph_choice != "entire":
            raise ValueError("Only graph='entire' is available when layer_mode='stack'.")
        if node_mode == "features" and graph_choice == "consensus":
            raise ValueError("graph='consensus' is not available when node_mode='features'; use 'entire' or a layer name.")
        if node_mode != "samples" and graph_choice == "fused":
            raise ValueError("graph='fused' is available only in multilayer samples mode.")
        if node_mode != "features" and graph_choice == "cross":
            raise ValueError("graph='cross' is available only in multilayer features mode.")
        if combine not in {"mean", "median", "max"}:
            raise ValueError("combine must be one of {'mean', 'median', 'max'}.")
        allowed_kwargs = {
            "clr": {"n_neighbors"},
            "rf": {"n_estimators", "max_depth"},
            "glasso": {"alpha", "max_iter", "tol"},
        }.get(method, set())
        unknown_kwargs = sorted(set(kwargs) - allowed_kwargs)
        if unknown_kwargs:
            names = ", ".join(map(str, unknown_kwargs))
            raise TypeError(f"Unexpected keyword arguments for method '{method}': {names}")
        infer_kwargs = kwargs.copy()
        sparsify_kwargs = dict(
            thr_raw=thr_raw,
            thr_norm=thr_norm,
            auto_target=auto_target,
            attach_isolates=attach_isolates,
            k=k,
            mutual=mutual,
        )

        if min_layers is None:
            min_layers = len(self.rodins) if (layer_mode == "multilayer" and node_mode == "samples") else 1
        min_layers = int(min_layers)
        if min_layers < 1:
            raise ValueError("min_layers must be >= 1.")
        if node_mode == "samples" and layer_mode == "multilayer" and min_layers > len(self.rodins):
            raise ValueError("min_layers cannot exceed the number of layers.")

        outputs = _build_outputs(
            self.rodins,
            self.names,
            node_mode=node_mode,
            layer_mode=layer_mode,
            method=method,
            n_jobs=n_jobs,
            infer_kwargs=infer_kwargs,
            sparsify_kwargs=sparsify_kwargs,
            combine=combine,
            min_layers=min_layers,
        )
        self._store_build_outputs(
            outputs,
            method=method,
            thr_raw=thr_raw,
            thr_norm=thr_norm,
            auto_target=auto_target,
            attach_isolates=attach_isolates,
            layer_mode=layer_mode,
            node_mode=node_mode,
            combine=combine,
            k=k,
            mutual=mutual,
            min_layers=min_layers,
            community_res=community_res,
        )
        self.set_graph(graph)
        _auto_sync_rodins(
            self,
            sample_graphs=(self.available_graphs(detailed=False) if node_mode == "samples" else ()),
        )

        if verbose:
            _print_build_summary(self)
        return self

    def adjust(
        self,
        *,
        graph: Optional[str] = None,
        thr_raw=_UNSET,
        thr_norm=_UNSET,
        auto_target=_UNSET,
        attach_isolates=_UNSET,
        k=_UNSET,
        mutual=_UNSET,
        min_layers=_UNSET,
        combine=_UNSET,
        community_res=_UNSET,
        verbose: bool = True,
    ) -> "Netan":
        """
        Rebuild graph variants from cached matrices without rerunning inference.

        Direct graphs are resparsified directly from their own matrices.
        Derived graphs are rebuilt through the support graphs they depend on,
        so adjusting ``entire`` or ``consensus`` can cascade to contributing
        layer graphs before rebuilding the requested derived graph.

        Parameters
        ----------
        graph : str, default=None
            Graph variant to update. If omitted, the current active graph is
            used.
        thr_raw : float, default=current setting
            Raw threshold on ``abs(W_raw)``.
        thr_norm : float, default=current setting
            Threshold on normalized similarity ``W_norm``.
        auto_target : float, default=current setting
            Automatic threshold target used when no manual threshold is active.
        attach_isolates : bool, default=current setting
            Reattach isolated nodes with one strongest edge when possible.
        k : int, 'auto', or None, default=current setting
            Optional kNN restriction after thresholding.
        mutual : bool, default=current setting
            Use mutual-kNN instead of union-kNN when k is active.
        min_layers : int, default=current setting
            Consensus support threshold when relevant.
        combine : {'mean', 'median', 'max'}, default=current setting
            Fusion rule for multilayer sample graphs.
        community_res : float, default=current setting
            Louvain resolution for module labels.
        verbose : bool, default=True
            If ``True``, print the compact graph summary after rebuilding.

        Notes
        -----
        Direct graphs (layer graphs, ``fused``, ``cross``, and stack
        ``entire``) are rebuilt directly from their cached matrices. Derived
        graphs (samples multilayer ``entire`` and ``consensus``, and features
        multilayer ``entire``) rebuild through the layer graphs they depend on.

        For derived graphs, ``thr_raw``, ``thr_norm``, ``auto_target``, ``k``,
        ``mutual``, and ``attach_isolates`` are applied to those contributing
        layer graphs first, then the requested derived graph is rebuilt from
        them. In samples mode, rebuilt graph labels are also synced
        automatically into each ``rodin.samples`` table under
        ``netan_<graph>_...`` columns. Graph-dependent cached scores, tuning
        results, and sample-graph rankings are invalidated.

        Methodology
        -----------
        ``k='auto'`` uses the same rule as ``build()``: no extra kNN is used
        when ``mean_degree_active <= 5``; otherwise
        ``k = 3 + floor(sqrt(mean_degree_active - 5))``, capped by
        ``ceil(median_degree_active)``, ``active_nodes - 1``, and ``10``.

        Returns
        -------
        Netan
            The same object with the selected graph family rebuilt.
        """
        if not self._graph_store:
            raise RuntimeError("Build the network first (.build).")

        meta = self._meta or {}
        matrix_cache = dict(self._cache.get("matrices", {}))
        node_mode = str(meta.get("nodeMode", "samples"))
        layer_mode = str(meta.get("layerMode", "stack"))
        tag_map = self._cache.get("tag_map") or {
            (str(name) or "layer").replace(".", "_"): str(name)
            for name in self.names
        }
        target_name, _ = self._resolve_active_graph(self._active_graph_name() if graph in (None, "") else str(graph))
        target_info = self._graph_info(target_name)
        combine_eff = (
            str(target_info.get("combine", meta.get("combine", "mean")))
            if combine is _UNSET
            else str(combine).lower()
        )
        if community_res is _UNSET:
            stored_community_res = target_info.get("community_res", meta.get("communityResolution", 1.0))
            try:
                community_res_eff = float(stored_community_res)
            except (TypeError, ValueError):
                community_res_eff = float(meta.get("communityResolution", 1.0))
        else:
            community_res_eff = float(community_res)
        if combine_eff not in {"mean", "median", "max"}:
            raise ValueError("combine must be one of {'mean', 'median', 'max'}.")
        if combine is not _UNSET and not (
            node_mode == "samples"
            and layer_mode == "multilayer"
            and target_name in {"fused", "entire", "consensus"}
        ):
            raise ValueError(
                "combine can be adjusted only for 'fused', 'entire', or 'consensus' "
                "in multilayer samples mode."
            )
        direct_kwargs = dict(
            thr_raw=thr_raw,
            thr_norm=thr_norm,
            auto_target=auto_target,
            attach_isolates=attach_isolates,
            k=k,
            mutual=mutual,
        )
        layer_names = [str(name) for name in self.G_layers]

        def feature_node_to_layer(ids: Sequence[str]) -> Dict[str, str]:
            return {
                node_id: tag_map.get(node_id.split("__", 1)[0], node_id.split("__", 1)[0])
                for node_id in map(str, ids)
                if "__" in node_id
            }

        def refresh_entries(
            graphs: Dict[str, nx.Graph],
            infos: Dict[str, Dict[str, Any]],
        ) -> None:
            _refresh_graph_entries(
                self,
                graphs,
                infos,
                node_mode=node_mode,
                community_res=community_res_eff,
            )

        if layer_mode == "stack":
            ids = list(map(str, self.G_all.nodes()))
            G_new, _, info_new = _rebuild_direct_graph(ids, matrix_cache["entire"], self._graph_info("entire"), **direct_kwargs)
            if node_mode == "samples":
                _set_single_layer_attrs(G_new, "entire")
            else:
                _annotate_feature_layers(G_new, feature_node_to_layer(ids), graph_label="entire")
            refresh_entries({"entire": G_new}, {"entire": info_new})
        elif target_name == "fused":
            ids = list(map(str, self._graph_obj("fused").nodes()))
            G_new, _, info_new = _rebuild_direct_graph(ids, matrix_cache["fused"], self._graph_info("fused"), **direct_kwargs)
            _annotate_support_layers(
                G_new,
                ids,
                {name: _graph_to_adjacency(ids, self._graph_obj(name)) for name in layer_names},
                graph_label="fused",
            )
            refresh_entries({"fused": G_new}, {"fused": info_new})
        elif node_mode == "samples":
            ids = list(map(str, self.sample_ids))
            target_layers = layer_names if target_name in {"entire", "consensus"} else [target_name]
            layer_graphs: Dict[str, nx.Graph] = {}
            layer_infos: Dict[str, Dict[str, Any]] = {}
            adj_by_layer: Dict[str, np.ndarray] = {}
            for layer_name in layer_names:
                if layer_name in target_layers:
                    G_layer, adj_layer, info_layer = _rebuild_direct_graph(
                        ids,
                        matrix_cache[layer_name],
                        self._graph_info(layer_name),
                        **direct_kwargs,
                    )
                    _set_single_layer_attrs(G_layer, layer_name)
                    layer_graphs[layer_name] = G_layer
                    layer_infos[layer_name] = info_layer
                    adj_by_layer[layer_name] = adj_layer
                else:
                    layer_graphs[layer_name] = self._graph_obj(layer_name)
                    layer_infos[layer_name] = self._graph_info(layer_name)
                    adj_by_layer[layer_name] = _graph_to_adjacency(ids, layer_graphs[layer_name])
            min_layers_eff = int(target_info.get("min_layers", meta.get("minLayers", 1))) if min_layers is _UNSET else int(min_layers)
            if min_layers_eff < 1:
                raise ValueError("min_layers must be >= 1.")
            if min_layers_eff > len(layer_names):
                raise ValueError("min_layers cannot exceed the number of layers.")
            derived = _samples_multilayer_derived(
                ids,
                {name: matrix_cache[name] for name in layer_names},
                layer_infos,
                adj_by_layer,
                combine=combine_eff,
                min_layers=min_layers_eff,
                sparsify_kwargs=_adjust_sparsify_kwargs(self._graph_info("fused"), **direct_kwargs),
            )
            graphs_to_update = {name: layer_graphs[name] for name in target_layers}
            infos_to_update = {name: layer_infos[name] for name in target_layers}
            graphs_to_update["entire"] = derived["graphs"]["entire"]
            infos_to_update["entire"] = derived["infos"]["entire"]
            graphs_to_update["consensus"] = derived["graphs"]["consensus"]
            infos_to_update["consensus"] = derived["infos"]["consensus"]
            refresh_entries(graphs_to_update, infos_to_update)
        else:
            support_targets = [*layer_names, "cross"] if target_name == "entire" else [target_name]
            layer_graphs: Dict[str, nx.Graph] = {}
            layer_infos: Dict[str, Dict[str, Any]] = {}
            for layer_name in layer_names:
                if layer_name in support_targets:
                    ids = list(map(str, self._graph_obj(layer_name).nodes()))
                    G_layer, _, info_layer = _rebuild_direct_graph(ids, matrix_cache[layer_name], self._graph_info(layer_name), **direct_kwargs)
                    _set_single_layer_attrs(G_layer, layer_name)
                    layer_graphs[layer_name] = G_layer
                    layer_infos[layer_name] = info_layer
                else:
                    layer_graphs[layer_name] = self._graph_obj(layer_name)
                    layer_infos[layer_name] = self._graph_info(layer_name)
            if "cross" in support_targets:
                ids = list(map(str, self._graph_obj("cross").nodes()))
                G_cross_new, _, info_cross = _rebuild_direct_graph(ids, matrix_cache["cross"], self._graph_info("cross"), **direct_kwargs)
                _set_single_layer_attrs(G_cross_new, "cross")
            else:
                G_cross_new = self._graph_obj("cross")
                info_cross = self._graph_info("cross")
            entire_ids = list(map(str, self._graph_obj("entire").nodes()))
            G_all_new, info_entire = _feature_entire_graph(
                entire_ids,
                layer_graphs,
                G_cross_new,
                W_abs=matrix_cache["entire"]["W_abs"],
                W_norm=matrix_cache["entire"]["W_norm"],
                base_info=_support_base_info([*layer_infos.values(), info_cross], fallback=info_cross),
            )
            graphs_to_update = {"entire": G_all_new}
            infos_to_update = {"entire": info_entire}
            if target_name == "entire":
                graphs_to_update.update(layer_graphs)
                infos_to_update.update(layer_infos)
                graphs_to_update["cross"] = G_cross_new
                infos_to_update["cross"] = info_cross
            elif target_name == "cross":
                graphs_to_update["cross"] = G_cross_new
                infos_to_update["cross"] = info_cross
            else:
                graphs_to_update[target_name] = layer_graphs[target_name]
                infos_to_update[target_name] = layer_infos[target_name]
            refresh_entries(graphs_to_update, infos_to_update)

        self._cache["matrices"] = matrix_cache
        self._cache["tag_map"] = tag_map

        self.set_graph(target_name)
        _invalidate_graph_analysis_results(self)
        _auto_sync_rodins(
            self,
            sample_graphs=(self.available_graphs(detailed=False) if node_mode == "samples" else ()),
        )
        if verbose:
            _print_build_summary(self)
        return self

    def info(self, *, verbose: bool = True) -> pd.DataFrame:
        """
        Return a summary table for all graph variants in the current object.

        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, also print the compact graph summary.

        Returns
        -------
        pandas.DataFrame
            One row per graph variant with graph type, build origin, and basic
            statistics.
        """
        if self.G_all is None:
            raise RuntimeError("Build the network first (.build).")

        out = _info_table(self)
        self._results()["info"] = out
        if verbose:
            _print_build_summary(self)
        return out

    def params(self, *, graph: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
        """
        Return the effective parameters of one graph variant.

        Parameters
        ----------
        graph : str, default=None
            Graph variant to inspect. If omitted, the current active graph is
            used.
        verbose : bool, default=True
            If ``True``, also print a compact parameter summary.

        Returns
        -------
        pandas.DataFrame
            One-row table with graph semantics and effective sparsification
            settings.
        """
        graph_name, _, _ = self._graph_context(graph)
        info = self._graph_info(graph_name)
        semantics = _graph_semantics(self._meta or {}, graph_name, layer_names=self.names)
        row = {
            "graph": graph_name,
            "method": (self._meta or {}).get("networkMethod"),
            "node_mode": (self._meta or {}).get("nodeMode"),
            **semantics,
            **_public_param_state(info, semantics, nt=self, graph_name=graph_name),
        }
        out = _trim_public_table(pd.DataFrame([row]))
        self._results()["params"] = out
        if verbose:
            print(
                "[Netan.params] "
                + " | ".join(
                    [
                        f"graph={graph_name}",
                        f"kind={semantics['kind']}",
                        f"method={row['method']}",
                        f"node_mode={row['node_mode']}",
                    ]
                )
            )
            param_parts = _format_public_param_parts(info, semantics, nt=self, graph_name=graph_name)
            line1 = [f"built_from={semantics['built_from']}"] + [part for part in param_parts if part.startswith("combine=")]
            line2 = [part for part in param_parts if part.startswith(("thr_", "auto"))]
            line3 = [part for part in param_parts if part.startswith("k")]
            used = set(line1[1:] + line2 + line3)
            line4 = [part for part in param_parts if part not in used]
            prefix = "  params: "
            pad = " " * len(prefix)
            print(prefix + " | ".join(line1))
            for group in (line2, line3, line4):
                if group:
                    print(pad + " | ".join(group))
        return out

    def samples(self, *, graph: Optional[str] = None) -> pd.DataFrame:
        """
        Return sample metadata for the selected graph.

        Adds graph-derived ``community`` and ``module`` columns to the sample
        table.

        Parameters
        ----------
        graph : str, default=None
            Graph variant to inspect. If omitted, the current active graph is
            used.

        Notes
        -----
        The returned table follows the sample order used by the selected graph,
        even if the original sample table is stored in a different row order.

        Returns
        -------
        pandas.DataFrame
            Sample table aligned to the graph.
        """
        if str((self._meta or {}).get("nodeMode", "samples")) != "samples":
            raise ValueError("samples() is available only when node_mode='samples'.")

        graph_name, G, _ = self._graph_context(graph)
        df = _aligned_samples_df(self)
        node_map = {
            str(node): {
                "community": attrs.get("community"),
                "module": attrs.get("module"),
            }
            for node, attrs in G.nodes(data=True)
        }
        out = df.copy()
        root = f"{_sync_prefix('netan')}_"
        managed = [
            c for c in map(str, out.columns)
            if c.startswith(root) and c.endswith(("_community", "_module"))
        ]
        if managed:
            out = out.drop(columns=managed)
        out["_sample_id"] = list(map(str, self.sample_ids))
        if "id" in out.columns:
            out["id"] = list(map(str, self.sample_ids))
        else:
            out.insert(0, "id", list(map(str, self.sample_ids)))
        out["community"] = out["_sample_id"].map(lambda x: node_map.get(x, {}).get("community"))
        out["module"] = out["_sample_id"].map(lambda x: node_map.get(x, {}).get("module"))
        out = out.drop(columns=["_sample_id"])
        self._results()["samples"] = out
        return out

    def nodes(
        self,
        *,
        graph: Optional[str] = None,
        active_only: bool = False,
    ) -> pd.DataFrame:
        """
        Return the node table for one graph variant.

        Parameters
        ----------
        graph : str, default=None
            Graph variant to inspect. If omitted, the current active graph is
            used.
        active_only : bool, default=False
            If ``True``, remove nodes with degree 0.

        Notes
        -----
        In samples mode this table is based on ``samples(graph=...)``. In
        features mode it is based on ``features()`` and automatically includes
        any currently available feature-ranking columns.

        Returns
        -------
        pandas.DataFrame
            Node table with graph attributes and merged sample or feature
            metadata.
        """
        graph_name, G, _ = self._graph_context(graph)
        node_rows = []
        for node, attrs in G.nodes(data=True):
            row = {"id": str(node)}
            row.update(attrs)
            node_rows.append(row)
        out = pd.DataFrame(node_rows) if node_rows else pd.DataFrame(columns=["id"])
        for col in ("x", "y"):
            if col not in out.columns:
                out[col] = None

        node_mode = str((self._meta or {}).get("nodeMode", "samples"))
        if node_mode == "samples":
            S = self.samples(graph=graph_name).copy()
            if "id" in S.columns:
                S["id"] = list(map(str, self.sample_ids))
            else:
                S.insert(0, "id", list(map(str, self.sample_ids)))
            S["id"] = S["id"].astype(str)
            out["id"] = out["id"].astype(str)
            extra = [c for c in S.columns if c not in {"id"} and c not in out.columns]
            out = out.merge(S[["id", *extra]], on="id", how="left")
        else:
            tag_map = self._cache.get("tag_map") or {
                (str(name) or "layer").replace(".", "_"): str(name)
                for name in self.names
            }
            feats = self.features().copy()
            split = out["id"].str.split("__", n=1, expand=True)
            if split.shape[1] == 2:
                out["_tag"] = split[0]
                out["feature_id"] = split[1].astype(str)
                out["layer"] = out["_tag"].map(tag_map).fillna(out["_tag"])
                merge_cols = ["layer", "feature_id"] + [
                    c for c in feats.columns if c not in {"layer", "feature_id"} and c not in out.columns
                ]
                out = out.merge(feats[merge_cols], on=["layer", "feature_id"], how="left")
                out = out.drop(columns=[c for c in ("_tag",) if c in out.columns])

        if active_only:
            out = out[out.get("degree", 0).fillna(0).astype(float) > 0].copy() if "degree" in out.columns else out

        self._results()["nodes"] = out
        return out

    def features(self) -> pd.DataFrame:
        """
        Return the current feature table.

        Notes
        -----
        Ranking columns are attached automatically. If both ``rank()`` and
        ``stability_rank()`` are available for the same graph and ``label``,
        the two tables are combined on ``layer + feature_id`` so the returned
        view exposes significance and stability columns together automatically.

        Returns
        -------
        pandas.DataFrame
            Feature metadata for the current inputs, optionally enriched with
            ranking columns.
        """
        feature_meta = self._cache.get("feature_meta")
        if not isinstance(feature_meta, dict) or not feature_meta:
            feature_meta = _feature_meta_lookup(self.names, self.rodins)
            self._cache["feature_meta"] = feature_meta
        rows = []
        managed_feature_cols = _managed_feature_sync_columns(prefix="netan")
        for layer_name, rodin in zip(map(str, self.names), self.rodins):
            tag = (layer_name or "layer").replace(".", "_")
            labels = (feature_meta.get(tag) or {}).get("labels", {})
            F = getattr(rodin, "features", None)
            if isinstance(F, pd.DataFrame) and not F.empty:
                base = _ensure_df(F, "r.features").copy()
                base.index = base.index.astype(str)
                managed = [c for c in map(str, base.columns) if c in managed_feature_cols]
                if managed:
                    base = base.drop(columns=managed)
            else:
                X = _ensure_df(getattr(rodin, "X", None), "r.X")
                base = pd.DataFrame(index=X.index.astype(str))
            base.insert(0, "feature", [labels.get(fid, fid) for fid in base.index.astype(str)])
            base.insert(0, "feature_id", base.index.astype(str))
            base.insert(0, "layer", layer_name)
            rows.append(base.reset_index(drop=True))
        out = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(columns=["layer", "feature_id", "feature"])
        ranked = _feature_results_bundle(self, require=False)["table"]
        if isinstance(ranked, pd.DataFrame) and not ranked.empty:
            keep = ["layer", "feature_id"] + [
                c for c in ranked.columns
                if c not in {"layer", "feature_id", "feature", "graph", "label"} and c not in out.columns
            ]
            out = out.merge(ranked[keep], on=["layer", "feature_id"], how="left")
        out = _feature_identity_index(out)
        self._results()["features"] = out
        return out

    def sync_samples_to_rodins(
        self,
        *,
        graph: Optional[str] = None,
        prefix: str = "netan",
        overwrite: bool = True,
    ) -> "Netan":
        """
        Write sample graph labels back into each underlying ``rodin.samples`` table.

        Parameters
        ----------
        graph : str, default=None
            Graph variant whose sample labels should be synced. If omitted, the
            current active graph is used.
        prefix : str, default='netan'
            Prefix used for the written columns.
        overwrite : bool, default=True
            If ``True``, replace previously synced columns with the same names.

        Returns
        -------
        Netan
            The same object with updated ``rodin.samples`` tables.

        Notes
        -----
        Standard Netan workflows sync these sample labels automatically after
        ``build()`` and ``adjust()`` in samples mode. Call this method only
        when you need a manual resync with custom settings.
        """
        if str((self._meta or {}).get("nodeMode", "samples")) != "samples":
            raise ValueError("sync_samples_to_rodins() requires a sample graph built with node_mode='samples'.")
        _sync_samples_back_to_rodins(self, graph=graph, prefix=prefix, overwrite=overwrite)
        _clear_sample_views(self)
        return self

    def sync_features_to_rodins(
        self,
        *,
        prefix: str = "netan",
        overwrite: bool = True,
    ) -> "Netan":
        """
        Write analytical ranking columns back into each ``rodin.features`` table.

        Parameters
        ----------
        prefix : str, default='netan'
            Prefix used for the written columns.
        overwrite : bool, default=True
            If ``True``, replace previously synced columns with the same names.

        Returns
        -------
        Netan
            The same object with updated ``rodin.features`` tables.

        Notes
        -----
        Standard Netan workflows sync ranking-derived feature columns
        automatically after ``rank()`` and ``stability_rank()``. Call this
        method only when you need a manual resync with custom settings.
        """
        if overwrite:
            _drop_managed_feature_sync_cols(self, prefix=prefix)
        _sync_features_back_to_rodins(self, prefix=prefix, overwrite=overwrite)
        _clear_feature_views(self)
        return self

    def scores(
        self,
        *,
        graph: Optional[str] = None,
        label: Optional[str] = None,
        weights: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Score one graph variant with the same metrics used by `tune()`.

        Parameters
        ----------
        graph : str, default=None
            Graph variant to score. If omitted, the current active graph is
            used.
        label : str, default=None
            Sample metadata column for supervised scoring in samples mode.
            Unique values are treated as class labels. Numeric columns are
            allowed and are not modeled as continuous targets.
        weights : dict, default=None
            Optional score-weight overrides. Blocks can be provided as
            sequences or dicts, for example ``{'sep': [15, 5, 80]}``,
            ``{'structure_supervised': [50, 35, 15]}``,
            ``{'structure_unsupervised': [40, 35, 25]}``, or
            ``{'supervised': [60, 15, 15, 10]}``. Supported blocks are
            ``structure_supervised``, ``structure_unsupervised``,
            ``stab_supervised``, ``stab_unsupervised``, ``sep``,
            ``supervised``, and ``unsupervised``. Missing blocks fall back to
            the library defaults.
        verbose : bool, default=True
            If ``True``, also print the compact score summary.

        Notes
        -----
        ``scores()`` uses the same final ``score`` that decides the winner in
        ``tune()`` and ``scores_grid()``. For the selected graph, local
        stability is estimated across the same refine-style neighborhood that
        ``tune()`` explores around the current configuration. When
        ``weights`` is omitted, the default library weights are used. The
        resulting one-row table is cached in ``self._results()['scores']``.

        Methodology
        -----------
        Unsupervised ``score`` is

        ``structure = 0.20 * modularity01 + 0.20 * degree_band``
        ``+ 0.60 * module_size_band``
        ``stab = 0.30 * module_stability + 0.70 * edge_stability``
        ``score = 0.50 * structure + 0.40 * stab + 0.10 * active_fraction``.

        Supervised ``score`` is

        ``structure = 0.20 * modularity01 + 0.20 * degree_band``
        ``+ 0.60 * module_size_band``
        ``sep = 0.15 * ari01 + 0.05 * nmi + 0.80 * assort01``
        ``stab = 0.30 * module_stability + 0.70 * edge_stability``
        ``score = 0.60 * sep + 0.15 * structure + 0.15 * stab``
        ``+ 0.10 * active_fraction``.

        Here ``ari01 = clip(max(ARI, 0), 0, 1)`` and
        ``assort01 = clip((label_assortativity + 1) / 2, 0, 1)``.

        Returns
        -------
        pandas.DataFrame
            One-row score table for the selected graph.
        """
        node_mode = str((self._meta or {}).get("nodeMode", "samples"))
        if node_mode == "features" and label is not None:
            raise ValueError("label is available only when node_mode='samples'.")
        objective = "supervised" if label is not None else "unsupervised"
        graph_name, _, _ = self._graph_context(graph)
        label_map = _sample_label_map(self, str(label)) if label is not None else None
        metrics = _score_current_graph(self, graph_name, objective=objective, label_map=label_map, weights=weights)
        public_metrics = _public_score_metrics(metrics, objective)
        out = _trim_public_table(
            pd.DataFrame(
                [
                    {
                        "objective": objective,
                        "graph": graph_name,
                        "method": (self._meta or {}).get("networkMethod"),
                        **public_metrics,
                    }
                ]
            )
        )
        self._results()["scores"] = out
        if verbose:
            print("[Netan.scores] " + " | ".join([f"objective={objective}", f"graph={graph_name}", f"method={(self._meta or {}).get('networkMethod', '-')}"]))
            _print_parts_block("stats", _format_stats_parts(metrics))
            _print_parts_block("scores", _format_score_parts(metrics, objective))
        return out

    def rank(
        self,
        label: str,
        *,
        graph: Optional[str] = None,
        layers: Optional[Union[str, Sequence[str]]] = None,
        use_weights: bool = True,
        standardize: bool = True,
        n_perm: int = 1000,
        seed: int = 1,
        fdr: bool = True,
        top: int = 10,
        chunk_size: int = 256,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Rank features by how well they follow label separation on a sample graph.

        Parameters
        ----------
        label : str
            Sample metadata column used as the target label. Unique values are
            treated as class labels, so numeric columns are allowed but are
            handled as discrete classes rather than continuous targets.
        graph : str, default=None
            Sample graph to use. If omitted, the current active graph is used.
        layers : str or sequence, default=None
            Optional subset of layers to rank.
        use_weights : bool, default=True
            Use edge weights when computing edge-based scores.
        standardize : bool, default=True
            Standardize feature values before scoring.
        n_perm : int, default=1000
            Number of label permutations for empirical p-values.
        seed : int, default=1
            Random seed.
        fdr : bool, default=True
            Compute FDR-adjusted p-values.
        top : int, default=10
            Number of top rows to print when verbose.
        chunk_size : int, default=256
            Internal chunk size for the feature loop.
        verbose : bool, default=True
            If ``True``, also print a compact ranking summary.

        Notes
        -----
        ``top_class`` is the class with the highest average feature value on
        labeled graph nodes. ``p_perm`` is the empirical permutation p-value
        for the edge-based score. ``p_adj`` is the Benjamini-Hochberg
        FDR-adjusted version of ``p_perm``. The latest rank result is also
        synced automatically back into each ``rodin.features`` table under
        ``netan_*`` analytical columns such as ``netan_rank``,
        ``netan_score``, and ``netan_p_adj``.

        Methodology
        -----------
        For each feature vector ``x``, Netan computes class-specific
        within-label dispersion ``W_c`` and pair-specific cross-label
        dispersion ``B_cd`` over graph edges using weighted squared
        differences ``w_ij * (x_i - x_j)^2``. The score is

        ``score = sum_{c<d} beta_cd * (log1p(B_cd) - 0.5 * (log1p(W_c) + log1p(W_d)))``

        where ``beta_cd`` are normalized pair weights proportional to
        ``sqrt(n_c * n_d * s_cd)`` where ``n_c`` is the number of labeled
        samples in class ``c`` and ``s_cd`` is the available edge support for
        the contrasted pair, defined as the minimum of the pair's cross-class
        support and the two classes' within-class support. In the binary case,
        this reduces to a single contrast between the two classes. High scores
        therefore mean larger variation across class boundaries together with
        smaller within-class variation in the contrasted classes. When a
        selected graph is too sparse to support any valid class pair with both
        cross-class edges and class-specific within-class support, Netan falls
        back to a pooled within-vs-cross statistic on the same weighted edge
        differences.
        ``p_perm`` is estimated by label
        permutation on the same score statistic, and ``p_adj`` is
        Benjamini-Hochberg FDR.

        Returns
        -------
        pandas.DataFrame
            Ranked feature table with score, permutation p-value, and adjusted
            p-value.
        """
        if str((self._meta or {}).get("nodeMode", "samples")) != "samples":
            raise ValueError("rank() requires a sample graph built with node_mode='samples'.")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty samples column name.")
        if isinstance(use_weights, bool) is False or isinstance(standardize, bool) is False or isinstance(fdr, bool) is False:
            raise TypeError("use_weights, standardize, and fdr must be bool values.")
        if not isinstance(n_perm, Integral) or int(n_perm) < 0:
            raise ValueError("n_perm must be an integer >= 0.")
        if not isinstance(seed, Integral):
            raise TypeError("seed must be an integer.")
        if not isinstance(chunk_size, Integral) or int(chunk_size) <= 0:
            raise ValueError("chunk_size must be an integer >= 1.")
        if not isinstance(top, Integral) or int(top) <= 0:
            raise ValueError("top must be an integer >= 1.")

        graph_name, G, _ = self._graph_context(graph)
        label_map = _sample_label_map(self, label)
        result = _rank_result(
            self,
            label=str(label),
            G=G,
            graph_name=graph_name,
            label_map=label_map,
            layers=layers,
            use_weights=bool(use_weights),
            standardize=bool(standardize),
            n_perm=int(n_perm),
            seed=int(seed),
            fdr=bool(fdr),
            chunk_size=int(chunk_size),
        )
        out = result["table"]
        _cohere_feature_ranking_results(self, keep="rank", graph_name=graph_name, label=str(label))
        self._results()["rank"] = result
        _auto_sync_rodins(self, sync_features=True)

        if verbose:
            print(
                "[Netan.rank] "
                + " | ".join(
                    [
                        f"graph={graph_name}",
                        f"label={label}",
                        f"layers={','.join(result['layers'])}",
                        f"features={len(out)}",
                        f"permutations={int(n_perm)}",
                        f"valid_permutations={int(result['valid_perm'])}",
                        f"weighted={bool(use_weights)}",
                        f"standardize={bool(standardize)}",
                    ]
                )
            )
            print(f"  edges: same={int(result['same_edges'])} | cross={int(result['cross_edges'])}")
            head = out.head(min(int(top), len(out)))
            print("  top:")
            for row in head.itertuples(index=False):
                print(
                    "    "
                    + " | ".join(
                        [
                            f"#{int(row.rank)}",
                            str(row.feature),
                            f"id={row.feature_id}",
                            f"class={row.top_class}",
                            f"score={float(row.score):.4f}",
                            f"p_perm={_fmt_opt(row.p_perm)}",
                            f"p_adj={_fmt_opt(row.p_adj)}",
                        ]
                    )
                )
        return out

    def stability_rank(
        self,
        label: str,
        *,
        graph: Optional[str] = None,
        layers: Optional[Union[str, Sequence[str]]] = None,
        sample_frac: float = 0.8,
        n_iter: int = 50,
        top: int = 20,
        stratify: bool = True,
        use_weights: bool = True,
        standardize: bool = True,
        seed: int = 1,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Estimate how stable the feature ranking is under repeated sample subsampling.

        Parameters
        ----------
        label : str
            Sample metadata column used as the target label. Unique values are
            treated as class labels, so numeric columns are allowed but are
            handled as discrete classes rather than continuous targets.
        graph : str, default=None
            Sample graph to use. If omitted, the current active graph is used.
        layers : str or sequence, default=None
            Optional subset of layers to rank.
        sample_frac : float, default=0.8
            Fraction of labeled samples to keep in each subsample.
        n_iter : int, default=50
            Number of subsampling iterations.
        top : int, default=20
            Number of top features counted as selected per iteration.
        stratify : bool, default=True
            Preserve class balance during subsampling when possible.
        use_weights : bool, default=True
            Use edge weights in the edge-based score.
        standardize : bool, default=True
            Standardize feature values before scoring.
        seed : int, default=1
            Random seed.
        verbose : bool, default=True
            If ``True``, also print a compact stability summary.

        Methodology
        -----------
        The graph is kept fixed. Each iteration samples a subset of labeled
        nodes, builds the induced subgraph, and reruns the same edge-based
        ranking used by ``rank()``. The final table aggregates selection
        frequency, mean rank, median rank, mean score, and score spread. The
        latest stability result is also synced automatically back into each
        ``rodin.features`` table under ``netan_*`` analytical columns such as
        ``netan_stability_rank`` and ``netan_selected_freq``.

        Returns
        -------
        pandas.DataFrame
            Stability table with selection frequency, mean rank, and mean
            score.
        """
        if str((self._meta or {}).get("nodeMode", "samples")) != "samples":
            raise ValueError("stability_rank() requires a sample graph built with node_mode='samples'.")
        if not isinstance(sample_frac, Real) or not (0 < float(sample_frac) <= 1):
            raise ValueError("sample_frac must be in (0, 1].")
        if not isinstance(n_iter, Integral) or int(n_iter) < 1:
            raise ValueError("n_iter must be an integer >= 1.")
        if not isinstance(top, Integral) or int(top) < 1:
            raise ValueError("top must be an integer >= 1.")

        graph_name, G, _ = self._graph_context(graph)
        label_map = _sample_label_map(self, label)
        labeled_nodes = [str(node) for node in G.nodes() if str(node) in label_map]
        if len(labeled_nodes) < 2:
            raise ValueError("stability_rank() requires at least two labeled nodes on the selected graph.")
        by_label: Dict[Any, List[str]] = {}
        for node in labeled_nodes:
            by_label.setdefault(label_map[node], []).append(node)
        if len(by_label) < 2:
            raise ValueError("label column must contain at least two classes on the selected graph.")

        selected_layers = list(map(str, self.names)) if layers is None else [str(x) for x in _grid_list(layers)]
        rng = np.random.default_rng(int(seed))
        runs: List[pd.DataFrame] = []
        valid_iter = 0
        for i in range(int(n_iter)):
            if stratify:
                chosen: List[str] = []
                for nodes in by_label.values():
                    take = min(len(nodes), max(1, int(np.ceil(float(sample_frac) * len(nodes)))))
                    chosen.extend(rng.choice(nodes, size=take, replace=False).tolist())
            else:
                take = min(len(labeled_nodes), max(2, int(np.ceil(float(sample_frac) * len(labeled_nodes)))))
                chosen = rng.choice(labeled_nodes, size=take, replace=False).tolist()
            H = G.subgraph(sorted(set(chosen))).copy()
            try:
                tab = _rank_result(
                    self,
                    label=str(label),
                    G=H,
                    graph_name=graph_name,
                    label_map=label_map,
                    layers=layers,
                    use_weights=bool(use_weights),
                    standardize=bool(standardize),
                    n_perm=0,
                    seed=int(seed) + i + 1,
                    fdr=False,
                    chunk_size=256,
                )["table"].copy()
            except Exception:
                continue
            tab["iter"] = valid_iter + 1
            tab["selected"] = False
            tab.loc[tab.index[: min(int(top), len(tab))], "selected"] = True
            runs.append(tab)
            valid_iter += 1
        if not runs:
            raise RuntimeError("stability_rank() failed for all resampling iterations.")

        stacked = pd.concat(runs, axis=0, ignore_index=True)
        agg = (
            stacked.groupby(["layer", "feature_id", "feature"], as_index=False)
            .agg(
                top_class=("top_class", lambda x: pd.Series(x).mode().iloc[0] if not pd.Series(x).mode().empty else str(x.iloc[0])),
                selected_freq=("selected", "mean"),
                mean_rank=("rank", "mean"),
                median_rank=("rank", "median"),
                score=("score", "mean"),
                score_sd=("score", "std"),
            )
        )
        agg["score_sd"] = agg["score_sd"].fillna(0.0)
        agg = agg.sort_values(["selected_freq", "mean_rank", "score"], ascending=[False, True, False]).reset_index(drop=True)
        agg.insert(0, "rank", np.arange(1, len(agg) + 1, dtype=int))
        agg = _feature_identity_index(agg)
        _cohere_feature_ranking_results(self, keep="rank_stability", graph_name=graph_name, label=str(label))
        self._results()["rank_stability"] = {
            "graph": graph_name,
            "label": str(label),
            "layers": selected_layers,
            "sample_frac": float(sample_frac),
            "n_iter": int(n_iter),
            "valid_iter": valid_iter,
            "top": int(top),
            "stratify": bool(stratify),
            "table": agg,
        }
        _auto_sync_rodins(self, sync_features=True)
        if verbose:
            print(
                "[Netan.stability_rank] "
                + " | ".join(
                    [
                        f"graph={graph_name}",
                        f"label={label}",
                        f"layers={','.join(selected_layers)}",
                        f"sample_frac={float(sample_frac):.3g}",
                        f"iterations={int(n_iter)}",
                        f"valid_iterations={valid_iter}",
                        f"top={int(top)}",
                        f"stratify={bool(stratify)}",
                    ]
                )
            )
            print("  top:")
            for row in agg.head(min(10, len(agg))).itertuples(index=False):
                print(
                    "    "
                    + " | ".join(
                        [
                            f"#{int(row.rank)}",
                            str(row.feature),
                            f"id={row.feature_id}",
                            f"freq={float(row.selected_freq):.4f}",
                            f"mean_rank={float(row.mean_rank):.3g}",
                            f"score={float(row.score):.4f}",
                        ]
                    )
                )
        return agg

    def best(self, *, verbose: bool = True) -> pd.DataFrame:
        """
        Return the current best result from the latest ``tune()`` or
        ``scores_grid()`` run.

        Parameters
        ----------
        verbose : bool, default=True
            If ``True``, also print a compact winner summary.

        Returns
        -------
        pandas.DataFrame
            One-row table with the winning graph, parameters, and score
            metrics.

        Notes
        -----
        ``best()`` reads the latest mirrored winner payload stored in
        ``self._results()['tune']`` by ``tune()`` or ``scores_grid()`` and
        returns it as a one-row public table.
        """
        tuned = self._results().get("tune") or {}
        table = tuned.get("table")
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise RuntimeError("Run tune() first before calling best().")

        build_params = dict(tuned.get("best_build_params") or {})
        adjust_params = dict(tuned.get("best_adjust_params") or {})
        metrics = dict(tuned.get("best_metrics") or {})
        row = {
            "objective": tuned.get("objective"),
            "method": tuned.get("best_method"),
            "layer_mode": tuned.get("best_layer_mode"),
            "graph": tuned.get("best_graph"),
            "score": tuned.get("score"),
            "applied": bool(tuned.get("applied", False)),
            **{k: v for k, v in build_params.items() if k not in {"method", "layer_mode", "graph", "n_jobs"}},
            **adjust_params,
            **metrics,
        }
        out = pd.DataFrame([row])
        self._results()["best"] = out
        if verbose:
            semantics = _graph_semantics(
                {
                    "nodeMode": row.get("node_mode") or build_params.get("node_mode"),
                    "layerMode": row.get("layer_mode"),
                },
                str(row.get("graph")),
                layer_names=self.names,
            )
            info = {
                "thr_raw": row.get("thr_raw"),
                "thr_norm": row.get("thr_norm"),
                "thr_raw_input": row.get("thr_raw_input"),
                "thr_norm_input": row.get("thr_norm_input"),
                "auto": row.get("auto"),
                "auto_target": row.get("auto_target"),
                "k_input": row.get("k_input"),
                "k": row.get("k"),
                "combine": row.get("combine"),
                "attach_isolates": row.get("attach_isolates"),
                "min_layers": row.get("min_layers"),
                "community_res": row.get("community_res"),
                "mutual": row.get("mutual"),
                "thr_raw_base": row.get("thr_raw_base"),
                "thr_norm_base": row.get("thr_norm_base"),
                "auto_base": row.get("auto_base"),
                "k_input_base": row.get("k_input_base"),
                "k_base": row.get("k_base"),
                "attach_isolates_base": row.get("attach_isolates_base"),
            }
            param_parts = [
                f"node_mode={row.get('node_mode') or build_params.get('node_mode')}",
                f"kind={semantics.get('kind', '-')}",
                f"built_from={semantics.get('built_from', '-')}",
                "\n",
                *_format_public_param_parts(
                    info,
                    semantics,
                    include_extras=True,
                    method_params=str(row.get("method_params") or "-"),
                ),
            ]
            tune_parts = [f"family={row.get('family') or '-'}"]
            if row.get("auto_target") is not None:
                tune_parts.append(f"auto_target={_fmt_opt(row.get('auto_target'))}")
            if row.get("thr_raw_input") is not None:
                tune_parts.append(f"thr_raw[input]={_fmt_opt(row.get('thr_raw_input'))}")
            if row.get("thr_norm_input") is not None:
                tune_parts.append(f"thr_norm[input]={_fmt_opt(row.get('thr_norm_input'))}")
            if row.get("k_input") is not None and row.get("k_input") != row.get("k"):
                tune_parts.append(f"k_input={row.get('k_input')}")
            if len(tune_parts) > 1 or tune_parts[0] != "family=-":
                param_parts.extend(["\n", *tune_parts])
            print(
                "[Netan.best] "
                + " | ".join(
                    [
                        f"objective={row.get('objective')}",
                        f"best={row.get('method')}:{row.get('layer_mode')}:{row.get('graph')}",
                        f"score={float(row.get('score', 0.0)):.4f}",
                        f"applied={bool(row.get('applied', False))}",
                    ]
                )
            )
            _print_parts_block(
                "search",
                [
                    f"coarse={int(tuned.get('coarse_candidates', 0))}",
                    f"final={int(tuned.get('final_candidates', 0))}",
                    f"failures={int(tuned.get('num_failures', 0))}",
                ],
            )
            _print_parts_block("params", param_parts)
            _print_parts_block("stats", _format_stats_parts(metrics))
            _print_parts_block("scores", _format_score_parts(metrics, str(row.get("objective") or "unsupervised")))
            weights = tuned.get("weights") or {}
            if isinstance(weights, dict):
                weight_parts = []
                for block in ("structure_supervised", "structure_unsupervised", "stab_supervised", "stab_unsupervised", "sep", "supervised", "unsupervised"):
                    block_weights = weights.get(block)
                    if isinstance(block_weights, dict) and block_weights:
                        weight_parts.append(
                            f"{block}="
                            + ", ".join(f"{name}:{_fmt_opt(val)}" for name, val in block_weights.items())
                        )
                if weight_parts:
                    _print_parts_block("weights", weight_parts)
        return out

    def shortlist(
        self,
        n: Optional[int] = None,
        *,
        layers: Optional[Union[str, Sequence[str]]] = None,
        per_layer: bool = False,
        p_adj_max: Optional[float] = None,
        p_max: Optional[float] = None,
        score_min: Optional[float] = None,
        rank_max: Optional[float] = None,
        selected_freq_min: Optional[float] = None,
        verbose: bool = True,
    ) -> "Netan":
        """
        Return a new container reduced to selected ranked features.

        Parameters
        ----------
        n : int, default=None
            Number of top rows to keep after filtering. If omitted, keep all
            rows that survive the other filters.
        layers : str or sequence, default=None
            Optional subset of layers to keep. If omitted, all layers present
            in the cached ranking result are eligible.
        per_layer : bool, default=False
            If ``True``, apply ``n`` separately within each layer.
        p_adj_max : float, default=None
            Keep only rows with ``p_adj <= p_adj_max`` when available.
        p_max : float, default=None
            Keep only rows with ``p_perm <= p_max`` when available.
        score_min : float, default=None
            Keep only rows with score at or above this value.
        rank_max : float, default=None
            Keep only rows with ``rank <= rank_max`` for ``rank`` results,
            and-or ``mean_rank <= rank_max`` for stability-based results.
        selected_freq_min : float, default=None
            Keep only rows with ``selected_freq >= selected_freq_min`` when
            available.
        verbose : bool, default=True
            If ``True``, also print a compact selection summary.

        Notes
        -----
        ``shortlist()`` automatically uses every currently available ranking
        column. If both ``rank()`` and ``stability_rank()`` exist for the same
        graph and ``label``, significance and stability columns are available
        together in one combined selection table.

        With defaults (``n=None``, ``layers=None``, and no threshold filters),
        the new object keeps every ranked feature currently present in the
        latest automatic ranking view. Sample rows are preserved as-is, while
        layers with no selected features are dropped.

        Returns
        -------
        Netan
            New container restricted to the selected ranked features.
        """
        if n is not None and (not isinstance(n, Integral) or int(n) <= 0):
            raise ValueError("n must be an integer >= 1 when provided.")

        bundle = _feature_results_bundle(self, require=True)
        rank_table = bundle["table"].copy()
        source_graph = str(bundle.get("graph") or self._active_graph_name())
        source_label = str(bundle.get("label") or "")

        if layers is not None:
            wanted = set(map(str, _grid_list(layers)))
            rank_table = rank_table[rank_table["layer"].astype(str).isin(wanted)].copy()
        if score_min is not None:
            score_col = "score" if "score" in rank_table.columns else ("stability_score" if "stability_score" in rank_table.columns else None)
            if score_col is not None:
                rank_table = rank_table[rank_table[score_col].astype(float) >= float(score_min)].copy()
        if p_adj_max is not None and "p_adj" in rank_table.columns:
            rank_table = rank_table[rank_table["p_adj"].fillna(np.inf).astype(float) <= float(p_adj_max)].copy()
        if p_max is not None and "p_perm" in rank_table.columns:
            rank_table = rank_table[rank_table["p_perm"].fillna(np.inf).astype(float) <= float(p_max)].copy()
        if rank_max is not None and "rank" in rank_table.columns:
            rank_table = rank_table[rank_table["rank"].astype(float) <= float(rank_max)].copy()
        if selected_freq_min is not None and "selected_freq" in rank_table.columns:
            rank_table = rank_table[rank_table["selected_freq"].astype(float) >= float(selected_freq_min)].copy()
        if rank_max is not None and "mean_rank" in rank_table.columns:
            rank_table = rank_table[rank_table["mean_rank"].astype(float) <= float(rank_max)].copy()
        if rank_table.empty:
            raise ValueError("No ranked features matched the requested layers.")

        picked = (
            rank_table.groupby("layer", group_keys=False, sort=False).head(int(n)).copy()
            if per_layer and n is not None
            else (rank_table.head(int(n)).copy() if n is not None else rank_table.copy())
        )
        if picked.empty:
            raise ValueError("No ranked features were selected.")

        feats_by_layer: Dict[str, List[str]] = {
            str(layer): grp["feature_id"].astype(str).tolist()
            for layer, grp in picked.groupby("layer", sort=False)
        }
        rodins_new, names_new = [], []
        for layer_name, rodin in zip(map(str, self.names), self.rodins):
            feat_ids = feats_by_layer.get(layer_name, [])
            if not feat_ids:
                continue
            rodins_new.append(_subset_rodin(rodin, features=feat_ids, samples=self.sample_ids))
            names_new.append(layer_name)
        if not rodins_new:
            raise ValueError("No layers remained after selecting features.")

        out = self.__class__(rodins=rodins_new, names=names_new, samples=list(self.sample_ids))
        picked = _feature_identity_index(picked)
        selected_keys = picked[["layer", "feature_id"]].copy()
        rank_cached = self._results().get("rank") or {}
        rank_base = rank_cached.get("table")
        if isinstance(rank_base, pd.DataFrame) and not rank_base.empty:
            rank_details = rank_cached.get("details")
            rank_payload = {
                **{k: rank_cached.get(k) for k in ("graph", "label", "layers", "use_weights", "standardize", "n_perm", "valid_perm", "same_edges", "cross_edges")},
                "table": _feature_identity_index(rank_base.merge(selected_keys, on=["layer", "feature_id"], how="inner")),
            }
            if isinstance(rank_details, pd.DataFrame):
                rank_payload["details"] = _feature_identity_index(
                    rank_details.merge(selected_keys, on=["layer", "feature_id"], how="inner")
                )
            out._results()["rank"] = rank_payload
        stab_cached = self._results().get("rank_stability") or {}
        stab_base = stab_cached.get("table")
        if isinstance(stab_base, pd.DataFrame) and not stab_base.empty:
            out._results()["rank_stability"] = {
                **{k: stab_cached.get(k) for k in ("graph", "label", "layers", "sample_frac", "n_iter", "valid_iter", "top", "stratify")},
                "table": _feature_identity_index(stab_base.merge(selected_keys, on=["layer", "feature_id"], how="inner")),
            }
        out._results()["shortlist"] = {
            "n": None if n is None else int(n),
            "per_layer": bool(per_layer),
            "graph": source_graph,
            "label": source_label,
            "table": picked,
        }
        if verbose:
            print(
                "[Netan.shortlist] "
                + " | ".join(
                    [
                        f"features={int(len(picked))}",
                        f"layers={','.join(names_new)}",
                        f"per_layer={bool(per_layer)}",
                        f"graph={source_graph}",
                        f"label={source_label}",
                    ]
                )
            )
        return out

    def edges(
        self,
        *,
        graph: Optional[str] = None,
        layer: Optional[str] = None,
        weight_min: Optional[float] = None,
        weight_max: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Return the edge table for one graph variant.

        Parameters
        ----------
        graph : str, default=None
            Graph variant to inspect. If omitted, the current active graph is
            used.
        layer : str, default=None
            Keep only edges whose layer labels include this value.
        weight_min : float, default=None
            Keep only edges with ``weight >= weight_min``.
        weight_max : float, default=None
            Keep only edges with ``weight <= weight_max``.

        Returns
        -------
        pandas.DataFrame
            Edge table for the selected graph.
        """
        graph_name, _, _ = self._graph_context(graph)
        entry = self._graph_store.get(str(graph_name), {})
        df = entry.get("edges")
        if df is None and entry.get("graph") is not None:
            df = _graph_edge_table(entry["graph"], (self._meta or {}).get("nodeMode", "samples"))
            entry["edges"] = df
        df = None if df is None else df.copy()
        if layer not in (None, ""):
            want = str(layer)
            df = df[
                df["layers"].apply(
                    lambda vals: want in vals
                    if isinstance(vals, (set, list, tuple))
                    else want in str(vals).split("|")
                )
            ].copy()
        if weight_min is not None:
            df = df[df["weight"] >= float(weight_min)].copy()
        if weight_max is not None:
            df = df[df["weight"] <= float(weight_max)].copy()
        return df

    def export(
        self,
        path: Optional[str] = None,
        *,
        graph: Optional[str] = None,
        sep: str = ",",
        index: bool = False,
        float_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Write an edge table to disk and return it.

        Parameters
        ----------
        path : str, default=None
            Output file path. If omitted, writes to ``./netan_<graph>.csv``.
        graph : str, default=None
            Graph variant to export. If omitted, the current active graph is
            used.
        sep : str, default=','
            CSV separator.
        index : bool, default=False
            Whether to write the DataFrame index.
        float_format : str, default=None
            Optional float formatting string for pandas.

        Returns
        -------
        pandas.DataFrame
            The same edge table that is written to disk.

        Notes
        -----
        The last written export target is cached in
        ``self._results()['last_export']`` with ``graph`` and ``path``.
        """
        graph_name, _, _ = self._graph_context(graph)
        df = self.edges(graph=graph_name)
        out_path = path or os.path.abspath(f"netan_{graph_name}.csv")
        df.to_csv(out_path, sep=sep, index=index, float_format=float_format)
        self._results()["last_export"] = {"graph": graph_name, "path": out_path}
        return df

    def grid(
        self,
        *,
        node_mode: Optional[str] = None,
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
        """
        Build the full tuning candidate grid once and return it.

        Parameters
        ----------
        node_mode : {'samples', 'features'}, default=current object
            Which node type to tune. If omitted, uses the current object
            ``node_mode`` when available, otherwise defaults to ``'samples'``.
        layer_modes, graphs, methods, method_grids, combine, auto_target, thr_norm, thr_raw, k, mutual, attach_isolates, min_layers
            Same search arguments accepted by ``tune()``.
        community_res : float, default=1.0
            Louvain resolution used while evaluating candidate graphs.
        verbose : bool, default=True
            Print build and refine progress.
        n_jobs : int, default=1
            Parallelism forwarded to inference methods.

        Notes
        -----
        ``grid()`` does the expensive part of ``tune()``: it builds base
        network states, applies the coarse adjust grid, runs the local refine
        neighborhood, and caches every final candidate together with its graph
        structure and unsupervised metrics. The full grid bundle is cached in
        ``self._cache['tune_grid']`` and the public summary table is cached in
        ``self._results()['grid']['table']``, so subsequent ``scores_grid()``
        and ``materialize()`` calls can omit ``grid``. Use ``scores_grid()``
        to score the same grid repeatedly with default or custom weights
        without rerunning inference.

        Returns
        -------
        dict
            Opaque tuning-grid bundle to pass into ``scores_grid()`` or
            ``materialize()``.
        """
        node_mode_eff = str((self._meta or {}).get("nodeMode", "samples")) if node_mode in (None, "") else str(node_mode)
        return _build_tune_grid(
            self,
            node_mode=node_mode_eff,
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
            community_res=community_res,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def scores_grid(
        self,
        grid: Optional[Dict[str, Any]] = None,
        *,
        label: Optional[str] = None,
        weights: Optional[Dict[str, Any]] = None,
        top_results: int = 10,
        apply: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Score a prebuilt tuning grid and return the ranked leaderboard.

        Parameters
        ----------
        grid : dict, default=None
            Bundle returned by ``grid()``. If omitted, uses the latest grid
            previously built on this object.
        label : str, default=None
            Sample metadata column for supervised scoring in samples mode.
            If omitted, the grid is scored unsupervised.
        weights : dict, default=None
            Optional score-weight overrides. Blocks can be passed as sequences
            or dicts. For example,
            ``{'sep': [15, 5, 80], 'supervised': [60, 15, 15, 10]}``
            overrides the supervised score, while
            ``{'structure_supervised': [50, 35, 15],``
            ``'structure_unsupervised': [40, 35, 25]}``
            overrides objective-specific structure blocks. Supported blocks
            are ``structure_supervised``, ``structure_unsupervised``,
            ``stab_supervised``, ``stab_unsupervised``, ``sep``,
            ``supervised``, and ``unsupervised``. Missing blocks fall back to
            the library defaults.
        top_results : int, default=10
            Number of rows returned in the public leaderboard.
        apply : bool, default=False
            Apply the winning candidate back to the current object.
        verbose : bool, default=True
            Print the winner summary.

        Notes
        -----
        ``scores_grid()`` reuses the already built candidate grid and only
        recomputes label-dependent terms plus the final weighted score. This
        makes weight iteration cheap compared with rebuilding the full search.
        When ``grid`` is omitted, the latest grid previously built on this
        object is used. The returned public leaderboard is also cached in
        ``self._results()['scores_grid']['table']``. The same latest winner
        payload is mirrored into ``self._results()['tune']`` so that
        ``best()`` can inspect the current winner consistently. The latest
        leaderboard is also stored on the grid itself in
        ``grid['last_scores']['leaderboard']`` so ``materialize(candidate=0)``
        can resolve zero-based leaderboard rows after a scoring run.

        Example
        -------
        ``nt.scores_grid(label='Source', weights={'sep': [15, 5, 80],``
        ``'structure_supervised': [50, 35, 15], 'structure_unsupervised':``
        ``[40, 35, 25], 'stab_supervised': [55, 45],``
        ``'stab_unsupervised': [70, 30], 'supervised': [60, 15, 15, 10],``
        ``'unsupervised': [50, 40, 10]})``
        rescales supervised ranking on the already built grid without
        rebuilding candidates.

        Returns
        -------
        pandas.DataFrame
            Ranked leaderboard of tuning candidates.
        """
        grid_bundle = _cached_tune_grid(self) if grid is None else grid
        return _score_tune_grid(
            self,
            grid_bundle,
            label=label,
            weights=weights,
            top_results=top_results,
            apply=apply,
            verbose=verbose,
        )

    def materialize(
        self,
        grid: Optional[Dict[str, Any]] = None,
        candidate: Optional[Any] = None,
        *,
        apply: bool = True,
    ) -> "Netan":
        """
        Materialize one grid candidate back into a live ``Netan`` object.

        Parameters
        ----------
        grid : dict, default=None
            Bundle returned by ``grid()``. If omitted, uses the latest grid
            previously built on this object.
        candidate : int or row-like, default=None
            Candidate selector. If omitted, materializes the latest winner from
            ``scores_grid()`` or ``tune()`` for this grid. When an integer is
            provided, it is first interpreted as a zero-based row index into
            the latest ``scores_grid()`` leaderboard for this grid. If that is
            not available or out of range, it is matched against
            ``candidate_id`` and then, if needed, treated as a zero-based
            positional index in the full grid. A leaderboard row also works.
            To force explicit ``candidate_id`` lookup, pass a dict such as
            ``{'candidate_id': 26}``.
        apply : bool, default=True
            If ``True``, apply the selected candidate back to the current
            object and return ``self``. If ``False``, return a detached
            ``Netan`` object containing the selected candidate state.

        Returns
        -------
        Netan
            The applied current object when ``apply=True``, otherwise a new
            object containing the selected built and adjusted graph state.

        Notes
        -----
        When ``grid`` is omitted, the latest grid previously built on this
        object is used. After ``scores_grid(apply=False)`` or
        ``tune(apply=False)``, calling ``materialize()`` with no explicit
        candidate applies the latest winner from that cached grid and cached
        leaderboard. Use ``materialize(candidate=0)`` for the first leaderboard
        row, ``materialize(candidate=1)`` for the second, and so on.
        """
        grid_bundle = _cached_tune_grid(self) if grid is None else grid
        candidate_nt = _materialize_grid_candidate(grid_bundle, candidate=candidate)
        if not apply:
            return candidate_nt
        return _apply_materialized_candidate(self, candidate_nt)

    def tune(
        self,
        *,
        node_mode: Optional[str] = None,
        label: Optional[str] = None,
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
        objective: str = "auto",
        weights: Optional[Dict[str, Any]] = None,
        community_res: float = 1.0,
        top_results: int = 10,
        apply: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Search build-time and adjust-time settings and return ranked candidates.

        Parameters
        ----------
        node_mode : {'samples', 'features'}, default=current object
            Which node type to tune. If omitted, uses the current object
            ``node_mode`` when available, otherwise defaults to ``'samples'``.
        label : str, default=None
            Sample metadata column for supervised tuning in samples mode.
            Unique values are treated as class labels. Numeric columns are
            allowed and are not modeled as continuous targets.
        layer_modes : str or sequence, default=['stack', 'multilayer']
            Layer modes to search, such as ``'stack'`` and ``'multilayer'``.
            When only one layer is available, the effective default is
            ``['stack']``.
        graphs : str or sequence, default=None
            Target graph variants to compare, such as ``'entire'``,
            ``'fused'``, ``'consensus'``, or ``'cross'`` when available.
            Default search targets are ``['entire', 'fused']`` for samples and
            ``['entire']`` for features. When only one layer is available, the
            effective default is always ``['entire']``.
        methods : sequence, default=['spearman', 'clr', 'rf']
            Network inference methods to evaluate.
        method_grids : dict, default=internal per-method grids
            Method-specific parameter grids, for example RF tree counts or CLR
            neighbor counts. Defaults are ``{}`` for ``spearman``,
            ``{'n_neighbors': [2, 5]}`` for ``clr``, and
            ``{'n_estimators': [160, 320], 'max_depth': [None, 8]}`` for
            ``rf``.
        combine : str or sequence, default=['mean']
            Fusion rules to search in multilayer samples mode.
        auto_target : float or sequence, default=[0.90, 0.94, 0.98]
            Automatic-threshold targets to evaluate. This is the only
            threshold family searched by default.
        thr_norm : float or sequence, default=[]
            Manual normalized thresholds to evaluate when requested
            explicitly.
        thr_raw : float or sequence, default=[]
            Manual raw thresholds to evaluate when requested explicitly.
        k : int, 'auto', None, or sequence, default=[None, 'auto']
            kNN settings to evaluate.
        mutual : bool or sequence, default=[False]
            Whether to evaluate mutual-kNN in coarse search. Refine may still
            try both ``False`` and ``True`` when ``k`` is active and the user
            did not pin ``mutual`` explicitly.
        attach_isolates : bool or sequence, default=[False]
            Whether to evaluate isolate reattachment.
        min_layers : int or sequence, default=[]
            Consensus support thresholds to evaluate when relevant.
        objective : {'auto', 'unsupervised', 'supervised'}, default='auto'
            Tuning objective.
        weights : dict, default=None
            Optional score-weight overrides forwarded to ``scores_grid()``.
            Blocks can be provided as sequences or dicts, for example
            ``{'sep': [15, 5, 80]}``, ``{'supervised': [60, 15, 15, 10]}``,
            or objective-specific blocks such as
            ``{'structure_supervised': [50, 35, 15]}`` and
            ``{'structure_unsupervised': [40, 35, 25]}``. Supported blocks are
            ``structure_supervised``, ``structure_unsupervised``,
            ``stab_supervised``, ``stab_unsupervised``, ``sep``,
            ``supervised``, and ``unsupervised``.
        community_res : float, default=1.0
            Louvain resolution used during evaluation.
        top_results : int, default=10
            Number of rows returned in the public leaderboard.
        apply : bool, default=True
            Apply the winning configuration back to the current object.
        verbose : bool, default=True
            Print progress and winner summaries.
        n_jobs : int, default=1
            Parallelism forwarded to inference methods.

        Notes
        -----
        ``method_grids`` should map each method to a parameter grid. Example::

            {'clr': {'n_neighbors': [2, 5]},
             'rf': {'n_estimators': [160, 320], 'max_depth': [None, 8]}}

        ``thr_raw``, ``thr_norm``, ``auto_target``, ``k``, ``mutual``, and
        ``attach_isolates`` can each be a single value or a sequence of values
        to search.

        In samples multilayer mode, ``graph='fused'`` is tuned directly, while
        ``graph='entire'`` and ``graph='consensus'`` are tuned through the
        layer graphs they are built from and then rescored after cascade
        rebuild.

        If the current object contains only one layer, including objects
        returned by ``shortlist()`` after one-layer filtering, ``tune()``
        restricts search to single-layer ``stack`` / ``entire`` candidates.

        Methodology
        -----------
        ``tune()`` is a convenience wrapper around ``grid()`` followed by
        ``scores_grid()``. Stage 1 first builds base network states across
        ``methods x method_grids x layer_modes x combine``. It then applies the
        coarse adjust grid on top of each built state using ``adjust()``, i.e.
        over ``auto_target``, ``thr_norm``, ``thr_raw``, ``k``, ``mutual``,
        ``attach_isolates``, and ``min_layers`` when relevant. Direct graphs
        are adjusted directly. Derived graphs are adjusted through the support
        graphs they depend on and then cascaded to the target graph.

        Stage 2 refines every stage-1 candidate locally without rerunning
        inference and scores each refined candidate with the same final score
        used by ``scores()``. For
        ``family='auto'``, refine uses
        ``auto_target in [center-0.02, center-0.01, center,``
        ``center+0.01, center+0.02]`` clipped to ``[1e-6, 1]``. For
        ``family='manual'``, refine uses
        ``thr_raw in [0.93, 0.97, 1.00, 1.03, 1.07] * center`` when
        ``thr_raw`` was active in the seed, and if ``thr_norm`` was active in
        the seed, it uses
        ``thr_norm in [center-0.02, center-0.01, center,``
        ``center+0.01, center+0.02]``. Refine also tries
        ``k`` from the seed request, ``None``, ``'auto'``, and when an
        effective kNN was active, ``k_eff-2``, ``k_eff-1``, ``k_eff``,
        ``k_eff+1``, and ``k_eff+2`` clipped to ``[2, 10]``. If ``mutual``
        was not pinned by the user,
        refine tries both ``False`` and ``True`` whenever ``k`` is active.
        For ``consensus``, refine keeps the seed ``min_layers`` fixed.

        Stage 2 ranks all refine candidates by the final ``score``. If
        ``weights`` is omitted, the default library weights are used.
        Unsupervised scoring is

        ``structure = 0.20 * modularity01 + 0.20 * degree_band``
        ``+ 0.60 * module_size_band``
        ``stab = 0.30 * module_stability + 0.70 * edge_stability``
        ``score = 0.50 * structure + 0.40 * stab + 0.10 * active_fraction``.

        Supervised scoring is

        ``structure = 0.20 * modularity01 + 0.20 * degree_band``
        ``+ 0.60 * module_size_band``
        ``sep = 0.15 * ari01 + 0.05 * nmi + 0.80 * assort01``
        ``stab = 0.30 * module_stability + 0.70 * edge_stability``
        ``score = 0.60 * sep + 0.15 * structure + 0.15 * stab``
        ``+ 0.10 * active_fraction``.

        Here ``ari01 = clip(max(ARI, 0), 0, 1)`` and
        ``assort01 = clip((label_assortativity + 1) / 2, 0, 1)``.
        ``scores()`` estimates ``module_stability`` and ``edge_stability``
        across the same nearby refine-style neighborhood around the current
        candidate, while ``tune()`` estimates them across the stage-2 local
        refine neighborhood already generated for that candidate family.
        The winning configuration is therefore both strong and locally stable.
        The returned leaderboard is the same object as
        ``self._results()['scores_grid']['table']``, the mirrored winner
        summary is stored in ``self._results()['tune']``, and after
        ``tune(apply=False)`` you can call ``materialize()`` or
        ``materialize(candidate=0)`` to apply a leaderboard row later.

        ``tune()`` caches the full candidate grid in
        ``self._cache['tune_grid']`` through ``grid()``. The returned
        leaderboard is cached in ``self._results()['scores_grid']['table']``
        and the latest winner payload is mirrored into ``self._results()['tune']``.
        If ``apply=False``, you can later call ``materialize()`` to apply the
        latest winner from ``tune()`` or ``materialize(candidate=0)`` to apply
        the first leaderboard row. With the default ``apply=True``, the winner
        is already applied to the current object.

        Returns
        -------
        pandas.DataFrame
            Ranked leaderboard of tuning candidates.
        """
        node_mode_eff = str((self._meta or {}).get("nodeMode", "samples")) if node_mode in (None, "") else str(node_mode)
        objective_eff = str(objective).lower()
        if node_mode_eff not in {"samples", "features"}:
            raise ValueError("node_mode must be 'samples' or 'features'.")
        if objective_eff not in {"auto", "unsupervised", "supervised"}:
            raise ValueError("objective must be 'auto', 'unsupervised', or 'supervised'.")
        if node_mode_eff == "features" and label is not None:
            raise ValueError("label is available only when node_mode='samples'.")
        label_eff = label
        if objective_eff == "unsupervised":
            label_eff = None
        elif objective_eff == "supervised":
            if label is None:
                raise ValueError("label must be provided for supervised tuning.")
            if node_mode_eff != "samples":
                raise ValueError("supervised tuning is available only in node_mode='samples'.")

        grid_bundle = self.grid(
            node_mode=node_mode_eff,
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
            community_res=community_res,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        return self.scores_grid(
            grid_bundle,
            label=label_eff,
            weights=weights,
            top_results=top_results,
            apply=apply,
            verbose=verbose,
        )

    def plot(
        self,
        *,
        graph: Optional[str] = None,
        color: Optional[str] = None,
        size: Optional[str] = None,
        shape: Optional[str] = None,
        layer: Optional[str] = None,
        hide_isolated: bool = False,
        weight_min: Optional[float] = None,
        weight_max: Optional[float] = None,
        node_size: int = 10,
        width: Optional[int] = None,
        height: Optional[int] = None,
        title: Optional[str] = None,
        continuous_colorscale: str = "Viridis",
        layout: str = "force-directed",
        layout_seed: int = 777
    ) -> "FigureWidget":
        """
        Render the selected graph as an interactive Plotly figure.

        Parameters
        ----------
        graph : str, default=None
            Graph variant to render. If omitted, the current active graph is
            used. This does not change the active graph stored on the object.
        color : str, default=None
            Node metadata column used for marker coloring.
        size : str, default=None
            Continuous node metadata column used for marker size scaling.
        shape : str, default=None
            Node metadata column used for marker shape.
        layer : str, default=None
            Restrict visible edges to a layer label inside the selected graph.
        hide_isolated : bool, default=False
            Hide isolated nodes after filtering.
        weight_min : float, default=None
            Minimum visible edge weight.
        weight_max : float, default=None
            Maximum visible edge weight.
        node_size : int, default=10
            Marker size.
        width : int, default=None
            Figure width in pixels.
        height : int, default=None
            Figure height in pixels.
        title : str, default=None
            Plot title.
        continuous_colorscale : str, default='Viridis'
            Plotly continuous colorscale.
        layout : str, default='force-directed'
            Layout name. Supported options are ``'force-directed'``,
            ``'spring'``, ``'circular'``, ``'kamada_kawai'``, and
            ``'random'``.
        layout_seed : int, default=777
            Seed for stochastic layouts.

        Notes
        -----
        ``size`` expects a continuous numeric node column. Categorical size
        inputs raise ``ValueError``.

        In features mode, ``color``, ``size``, and ``shape`` can use any
        ranking columns currently available through ``features()``.

        ``layer`` filters edges inside the selected graph. It does not switch
        graph variants. Use ``graph=...`` to render another stored graph
        without changing the active graph on the object.

        If ``color``, ``size``, and ``shape`` are all omitted, the plot hides
        the legend. With continuous ``color`` and no ``shape``, the plot shows
        a colorbar instead of a categorical legend. The returned figure is
        also stored in ``self.fig``.

        Returns
        -------
        plotly.graph_objects.FigureWidget
            Interactive Plotly graph.
        """
        from .plotting import plot_netan

        return plot_netan(
            self,
            graph=graph,
            color=color,
            size=size,
            shape=shape,
            layer=layer,
            hide_isolated=hide_isolated,
            weight_min=weight_min,
            weight_max=weight_max,
            node_size=node_size,
            width=width,
            height=height,
            title=title,
            continuous_colorscale=continuous_colorscale,
            layout=layout,
            layout_seed=layout_seed,
        )

    def to_csv(
        self,
        path: Optional[str] = None,
        *,
        graph: Optional[str] = None,
        sep: str = ",",
        index: bool = False,
        float_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        CSV helper.

        Parameters
        ----------
        path : str, default=None
            If omitted, return the edge table without writing. Otherwise
            delegate to ``export()`` and write it to disk.
        graph : str, default=None
            Graph variant to export. If omitted, the current active graph is
            used.
        sep : str, default=','
            CSV separator.
        index : bool, default=False
            Whether to write the DataFrame index.
        float_format : str, default=None
            Optional float formatting string for pandas.

        Returns
        -------
        pandas.DataFrame
            Edge table for the selected graph.
        """
        if path is None:
            return self.edges(graph=graph)
        return self.export(
            path,
            graph=graph,
            sep=sep,
            index=index,
            float_format=float_format,
        )

    def save(self, path: str) -> str:
        """
        Serialize the current object to disk with pickle.

        Parameters
        ----------
        path : str
            Output pickle path.

        Returns
        -------
        str
            Absolute path written to disk.
        """
        out_path = os.path.abspath(str(path))
        fig = self.fig
        self.fig = None
        try:
            with open(out_path, "wb") as fh:
                pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.fig = fig
        return out_path

_DEFAULT_TUNE_METHOD_GRIDS: Dict[str, Dict[str, Sequence[Any]]] = {
    "spearman": {},
    "clr": {"n_neighbors": [2, 5]},
    "rf": {"n_estimators": [160, 320], "max_depth": [None, 8]},
}

_DEFAULT_TUNE_ADJUST_GRID: Dict[str, Sequence[Any]] = {
    "auto_target": [0.90, 0.94, 0.98],
    "thr_norm": [],
    "thr_raw": [],
    "k": [None, "auto"],
    "mutual": [False],
    "attach_isolates": [False],
    "min_layers": [],
}

_VALID_TUNE_GRAPHS: Dict[str, Dict[str, List[str]]] = {
    "samples": {"stack": ["entire"], "multilayer": ["entire", "fused", "consensus"]},
    "features": {"stack": ["entire"], "multilayer": ["entire", "cross"]},
}

def _grid_list(value: Any) -> List[Any]:
    if value is None:
        return [None]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


from ._views import (
    _aligned_samples_df,
    _auto_sync_rodins,
    _clear_feature_views,
    _clear_sample_views,
    _cohere_feature_ranking_results,
    _feature_identity_index,
    _feature_results_bundle,
    _fmt_opt,
    _format_public_param_parts,
    _format_score_parts,
    _format_stats_parts,
    _graph_semantics,
    _info_table,
    _invalidate_graph_analysis_results,
    _managed_feature_sync_columns,
    _drop_managed_feature_sync_cols,
    _print_build_summary,
    _print_parts_block,
    _public_param_state,
    _public_score_metrics,
    _sync_features_back_to_rodins,
    _sync_prefix,
    _sync_samples_back_to_rodins,
    _trim_public_table,
)
from ._ranking import (
    _rank_result_impl as _rank_result,
)
from ._tuning import (
    _apply_materialized_candidate_impl as _apply_materialized_candidate,
    _build_tune_grid_impl as _build_tune_grid,
    _cached_tune_grid_impl as _cached_tune_grid,
    _materialize_grid_candidate_impl as _materialize_grid_candidate,
    _sample_label_map_impl as _sample_label_map,
    _score_current_graph_impl as _score_current_graph,
    _score_tune_grid_impl as _score_tune_grid,
)

def create(
    rodins: Union[object, Sequence[object]],
    names: Optional[Sequence[str]] = None,
) -> "Netan":
    """
    Create a Netan container from one or more Rodin-like inputs.

    This is the main entry point for a Netan workflow. Inputs are aligned to
    their shared samples, layer names are assigned, and a ready-to-build
    container is returned.

    Parameters
    ----------
    rodins : object or sequence
        One Rodin-like object or a sequence of them. Each object should expose
        ``X`` and ``samples``. Feature metadata in ``features`` and layer
        metadata in ``uns`` are used when available.
    names : sequence, default=None
        Layer names. If omitted, Netan assigns ``L1``, ``L2``, and so on.
        Reserved graph names ``entire``, ``fused``, ``consensus``, and
        ``cross`` are not allowed as layer names.

    Notes
    -----
    Only samples shared by all inputs are kept. Sample order follows the first
    input after intersection. Use one object for one-layer analyses or a
    sequence for multilayer analyses.

    Returns
    -------
    Netan
        Container with aligned inputs, layer names, and shared samples.
    """
    objs = [rodins] if not isinstance(rodins, (list, tuple)) else list(rodins)
    if not objs:
        raise ValueError("Provide at least one Rodin-like object.")

    orig_ids_list, orig_shapes = [], []
    for r in objs:
        try:
            samples_df = _ensure_df(getattr(r, "samples", None), "r.samples")
            orig_ids_list.append(_extract_sample_ids_from_df(samples_df) or [])
        except Exception:
            orig_ids_list.append([])
        orig_shapes.append(tuple(getattr(r, "X").shape) if getattr(r, "X", None) is not None else (None, None))

    ids = common_samples(objs)
    if not ids:
        raise ValueError("No common samples across provided objects.")

    excluded_per_obj = []
    ids_set = set(ids)
    for orig_ids in orig_ids_list:
        excluded_per_obj.append([s for s in orig_ids if s not in ids_set])

    objs = [_subset_rodin(r, samples=ids) for r in objs]
    final_shapes = [tuple(getattr(r, "X").shape) if getattr(r, "X", None) is not None else (None, None) for r in objs]

    if names is None:
        names = [f"L{i}" for i in range(1, len(objs) + 1)]
    else:
        if len(names) != len(objs):
            raise ValueError("Length of 'names' must match number of rodin objects.")
        names = list(map(str, names))
    names = _validate_layer_names(names)

    def _preview(lst, n=8):
        if not lst:
            return "-"
        return ", ".join(lst[:n]) + (f", … (+{len(lst)-n})" if len(lst) > n else "")

    if len(objs) > 1:
        print(f"[Netan] common samples: {len(ids)} -> {_preview(ids)}")
        for nm, os, fs, excl in zip(names, orig_shapes, final_shapes, excluded_per_obj):
            os_str = f"{os[0]}x{os[1]}" if os[0] is not None else "N/A"
            fs_str = f"{fs[0]}x{fs[1]}" if fs[0] is not None else "N/A"
            if excl:
                print(f"[Netan] {nm}: X {os_str} -> {fs_str}; dropped {len(excl)}: {_preview(excl)}")
            else:
                print(f"[Netan] {nm}: X {os_str}")
    else:
        print(f"[Netan] {names[0]}: X {objs[0].X.shape[0]}x{objs[0].X.shape[1]}")

    return Netan(rodins=objs, names=names, samples=ids)


def load(path: str) -> "Netan":
    """
    Restore a pickled ``Netan`` object from disk.

    Parameters
    ----------
    path : str
        Input pickle path.

    Returns
    -------
    Netan
        Restored object.
    """
    with open(os.path.abspath(str(path)), "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, Netan):
        raise TypeError("Pickle does not contain a Netan object.")
    return obj
