"""Internal view, sync, and formatting helpers for Netan."""

import io
import re
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import netan as _n
from ._build import _display_k, _ensure_df, _extract_sample_ids_from_df, _requested_k

_FEATURE_SYNC_COLUMNS = _n._FEATURE_SYNC_COLUMNS
_CORE_GRAPHS = _n._CORE_GRAPHS

def _graph_semantics(
    meta: Dict[str, Any],
    graph_name: str,
    *,
    layer_names: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    layer_text = ",".join(map(str, layer_names or []))
    layers_built_from = f"layers[{layer_text}]" if layer_text else "layers"
    node_mode = str(meta.get("nodeMode", "samples"))
    layer_mode = str(meta.get("layerMode", "stack"))
    if node_mode == "samples":
        if layer_mode == "multilayer":
            if graph_name == "fused":
                return {"kind": "direct", "built_from": "fused_matrix"}
            if graph_name in {"entire", "consensus"}:
                return {"kind": "derived", "built_from": layers_built_from}
            return {"kind": "direct", "built_from": "layer_matrix"}
        return {"kind": "direct", "built_from": "matrix"}
    if layer_mode == "multilayer":
        if graph_name == "entire":
            return {"kind": "derived", "built_from": f"{layers_built_from}+cross"}
        if graph_name == "cross":
            return {"kind": "direct", "built_from": "cross_matrix"}
        return {"kind": "direct", "built_from": "layer_matrix"}
    return {"kind": "direct", "built_from": "matrix"}

def _public_param_state(
    info: Dict[str, Any],
    semantics: Dict[str, str],
    *,
    nt: Optional["Netan"] = None,
    graph_name: Optional[str] = None,
) -> Dict[str, Any]:
    if semantics.get("kind") == "derived":
        out = _derived_base_param_state(nt, str(graph_name)) if nt is not None and graph_name is not None else {}
        out.setdefault("attach_isolates_base", bool(info.get("attach_isolates", False)))
        out.setdefault("k_input_base", _requested_k(info))
        return {**out, "combine": info.get("combine"), "min_layers": info.get("min_layers"), "community_res": info.get("community_res")}
    out = {
        "thr_raw": info.get("thr_raw"),
        "thr_norm": info.get("thr_norm"),
        "auto": (info.get("auto_info") or {}).get("strategy") or info.get("auto"),
        "k_input": _requested_k(info),
        "k": _display_k(info),
        "combine": info.get("combine"),
        "attach_isolates": bool(info.get("attach_isolates", False)),
        "min_layers": info.get("min_layers"),
        "community_res": info.get("community_res"),
        "mutual": bool(info.get("mutual", False)),
    }
    return out

def _collapse_base_value(
    names: Sequence[str],
    values: Sequence[Any],
    *,
    compare=None,
    display=None,
) -> Any:
    names = list(map(str, names))
    vals = list(values)
    if not vals:
        return None
    compare = compare or (lambda x: x)
    display = display or compare
    normalized = [compare(v) for v in vals]
    non_null = [v for v in normalized if v is not None]
    if not non_null:
        return None
    if len(set(non_null)) == 1 and len(non_null) == len(normalized):
        return vals[normalized.index(non_null[0])]
    items = []
    for name, value in zip(names, vals):
        shown_value = display(value)
        items.append(f"{name}={shown_value if shown_value is not None else '-'}")
    return "{" + ", ".join(items) + "}"

def _derived_base_param_state(nt: "Netan", graph_name: str) -> Dict[str, Any]:
    meta = nt._meta or {}
    node_mode = str(meta.get("nodeMode", "samples"))
    layer_mode = str(meta.get("layerMode", "stack"))
    names = list(map(str, nt.G_layers))
    if not (
        (node_mode == "samples" and layer_mode == "multilayer" and graph_name in {"entire", "consensus"})
        or (node_mode == "features" and layer_mode == "multilayer" and graph_name == "entire")
    ):
        names = []
    elif node_mode == "features" and nt._graph_obj("cross") is not None:
        names.append("cross")
    selected = [(name, nt._graph_info(name)) for name in names if nt._graph_obj(name) is not None]
    names = [name for name, _ in selected]
    infos = [info for _, info in selected]
    if not infos:
        return {}
    return {
        "thr_raw_base": _collapse_base_value(names, [info.get("thr_raw") for info in infos], compare=_fmt_opt, display=_fmt_opt),
        "thr_norm_base": _collapse_base_value(names, [info.get("thr_norm") for info in infos], compare=_fmt_opt, display=_fmt_opt),
        "auto_base": _collapse_base_value(
            names,
            [(info.get("auto_info") or {}).get("strategy") or info.get("auto") for info in infos],
            compare=str,
            display=str,
        ),
        "k_input_base": _collapse_base_value(
            names,
            [_requested_k(info) for info in infos],
            compare=lambda v: "None" if v is None else str(v),
            display=lambda v: "None" if v is None else str(v),
        ),
        "k_base": _collapse_base_value(names, [_display_k(info) for info in infos], compare=str, display=str),
        "attach_isolates_base": _collapse_base_value(
            names,
            [bool(info.get("attach_isolates", False)) for info in infos],
            compare=str,
            display=str,
        ),
    }
def _fmt_opt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    val = float(x)
    return "-" if not np.isfinite(val) else f"{val:.4g}"

def _graph_detail_parts(stats: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
    parts = [
        f"communities={stats['numCommunities']}",
        f"modules={stats['numModules']}",
        f"isolates={int(info.get('isolate_count', 0))}",
        f"density_all={stats['densityAll']}",
        f"density_active={stats['densityActive']}",
    ]
    if info.get("used_isolate_reattachment"):
        parts.append(f"reattached={int(info.get('isolates_reattached', 0))}")
    return parts

def _is_direct_thresholded_graph(name: str, meta: Dict[str, Any]) -> bool:
    if name == "consensus":
        return False
    if name == "entire" and meta.get("layerMode") == "multilayer":
        return False
    return True

def _print_build_summary(nt: "Netan"):
    meta = nt._meta or {}
    active = nt._active_graph_name()
    active_info = nt._graph_info(active)
    active_semantics = _graph_semantics(meta, active)
    header_parts = [
        f"method={meta.get('networkMethod')}",
        f"node_mode={meta.get('nodeMode')}",
        f"layer_mode={meta.get('layerMode')}",
        f"active={active}",
        f"kind={active_semantics['kind']}",
    ]
    if _is_direct_thresholded_graph(active, meta):
        header_parts.extend(
            [
                f"thr_raw={_fmt_opt(active_info.get('thr_raw'))}",
                f"thr_norm={_fmt_opt(active_info.get('thr_norm'))}",
                f"auto={(active_info.get('auto_info') or {}).get('strategy') or active_info.get('auto') or '-'}",
                f"k={_display_k(active_info)}",
            ]
        )
    print("[Netan] " + " | ".join(header_parts))

    order = [active, *_CORE_GRAPHS, *[name for name in nt._graph_store if name not in _CORE_GRAPHS]]
    seen = set()
    for name in order:
        stats = nt._graph_stats(name)
        if name in seen or not stats:
            continue
        seen.add(name)
        info = nt._graph_info(name)
        semantics = _graph_semantics(meta, name)
        parts = [
            f"kind={semantics['kind']}",
            f"nodes={stats['numNodes']}",
            f"edges={stats['numEdges']}",
            f"active_nodes={stats['nodesWithEdges']}",
        ]
        if _is_direct_thresholded_graph(name, meta):
            parts.extend(
                [
                    f"thr_raw={_fmt_opt(info.get('thr_raw'))}",
                    f"thr_norm={_fmt_opt(info.get('thr_norm'))}",
                ]
            )
            if info.get("auto_info"):
                auto = info["auto_info"]
                parts.append(f"auto={auto.get('strategy')}")
            parts.append(f"k={_display_k(info)}")
        if info.get("min_layers"):
            parts.append(f"min_layers={info['min_layers']}")
        print(f"  [{name}] " + " | ".join(parts))
        print("    " + " | ".join(_graph_detail_parts(stats, info)))
        print()

    if meta.get("nodeMode") == "features":
        cross_stats = (nt._graph_stats("entire") or {}).get("layerStats", {}).get("cross")
        if cross_stats and "cross" not in seen:
            if meta.get("layerMode") == "multilayer" and nt._graph_stats("cross"):
                cross_info = nt._graph_info("cross")
                cross_net = nt._graph_stats("cross")
                parts = [
                    f"nodes={cross_net['numNodes']}",
                    f"edges={cross_net['numEdges']}",
                    f"active_nodes={cross_net['nodesWithEdges']}",
                    f"thr_raw={_fmt_opt(cross_info.get('thr_raw'))}",
                    f"thr_norm={_fmt_opt(cross_info.get('thr_norm'))}",
                ]
                if cross_info.get("auto_info"):
                    auto = cross_info["auto_info"]
                    parts.append(f"auto={auto.get('strategy')}")
                parts.append(f"k={_display_k(cross_info)}")
                print("  [cross] " + " | ".join(parts))
                print("    " + " | ".join(_graph_detail_parts(cross_net, cross_info)))
            else:
                print(
                    "  [cross] "
                    + " | ".join(
                        [
                            f"nodes={int(cross_stats.get('nodes', 0))}",
                            f"edges={int(cross_stats.get('edges', 0))}",
                            f"density={float(cross_stats.get('density', 0.0)):.4g}",
                        ]
                    )
                )
            print()


def _silent(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)

def _aligned_samples_df(nt: "Netan") -> pd.DataFrame:
    cached = nt._cache.get("aligned_samples_df")
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    samples_df = _ensure_df(getattr(nt.rodins[0], "samples", None), "r.samples").copy()
    sample_ids = list(map(str, nt.sample_ids))
    if len(samples_df) != len(sample_ids):
        raise ValueError("r.samples is not aligned to the current sample order.")

    detected_ids = _extract_sample_ids_from_df(samples_df, sample_ids)
    if detected_ids is not None:
        aligned = samples_df.copy()
        aligned["_sample_id"] = detected_ids
        aligned = aligned.set_index("_sample_id").loc[sample_ids].reset_index(drop=True)
        nt._cache["aligned_samples_df"] = aligned.copy()
        return aligned

    aligned = samples_df.reset_index(drop=True)
    nt._cache["aligned_samples_df"] = aligned.copy()
    return aligned

def _info_table(nt: "Netan") -> pd.DataFrame:
    rows = []
    for name in [key for key in _CORE_GRAPHS if nt._graph_obj(key) is not None] + [name for name in nt._graph_store if name not in _CORE_GRAPHS and nt._graph_obj(name) is not None]:
        info = nt._graph_info(name)
        stats = nt._graph_stats(name)
        rows.append(
            {
                "graph": name,
                **_graph_semantics(nt._meta or {}, name, layer_names=nt.names),
                "nodes": int(stats.get("numNodes", 0)),
                "edges": int(stats.get("numEdges", 0)),
                "active_nodes": int(stats.get("nodesWithEdges", 0)),
                "communities": int(stats.get("numCommunities", 0)),
                "modules": int(stats.get("numModules", 0)),
                "isolates": int(info.get("isolate_count", max(int(stats.get("numNodes", 0)) - int(stats.get("nodesWithEdges", 0)), 0))),
                "density_all": float(stats.get("densityAll", 0.0)),
                "density_active": float(stats.get("densityActive", 0.0)),
            }
        )
    return _trim_public_table(pd.DataFrame(rows))

def _trim_public_table(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    keep = [c for c in df.columns if not df[c].isna().all()]
    return df.loc[:, keep].copy()

def _managed_feature_sync_columns(
    *,
    prefix: str = "netan",
) -> set:
    root = _sync_prefix(prefix)
    return {f"{root}_{col}" for col in _FEATURE_SYNC_COLUMNS}

def _feature_identity_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if not {"layer", "feature_id"}.issubset(df.columns):
        return df
    out = df.copy()
    out["layer"] = out["layer"].astype(str)
    out["feature_id"] = out["feature_id"].astype(str)
    out.index = pd.MultiIndex.from_arrays(
        [out["layer"].to_numpy(), out["feature_id"].to_numpy()],
        names=["layer_idx", "feature_id_idx"],
    )
    return out

def _sync_prefix(prefix: str) -> str:
    token = str(prefix).strip()
    if not token:
        raise ValueError("prefix must be a non-empty string.")
    return token.strip("_")

def _stability_feature_view(table: pd.DataFrame, *, rename_feature: bool) -> pd.DataFrame:
    rename = {
        "rank": "stability_rank",
        "top_class": "stability_top_class",
        "score": "stability_score",
        "score_sd": "stability_score_sd",
    }
    if rename_feature:
        rename["feature"] = "stability_feature"
    return table.rename(columns=rename).copy()


def _feature_results_bundle(nt: "Netan", *, require: bool = False) -> Dict[str, Any]:
    rank_cached = nt._results().get("rank") or {}
    stab_cached = nt._results().get("rank_stability") or {}
    rank_table = rank_cached.get("table")
    stab_table = stab_cached.get("table")
    has_rank = isinstance(rank_table, pd.DataFrame) and not rank_table.empty
    has_stability = isinstance(stab_table, pd.DataFrame) and not stab_table.empty

    if not has_rank and not has_stability:
        if require:
            raise RuntimeError("Run rank() or stability_rank() first before requesting ranked features.")
        return {
            "table": pd.DataFrame(),
            "graph": None,
            "label": None,
        }

    if has_rank and has_stability:
        rank_graph = rank_cached.get("graph")
        stab_graph = stab_cached.get("graph")
        rank_label = rank_cached.get("label")
        stab_label = stab_cached.get("label")
        if rank_graph != stab_graph or rank_label != stab_label:
            raise RuntimeError(
                "Current rank() and stability_rank() results belong to different "
                "graph/label combinations. Rerun them on the same graph and label "
                "before requesting combined ranked features."
            )
        merged = rank_table.merge(
            _stability_feature_view(stab_table, rename_feature=True),
            on=["layer", "feature_id"],
            how="outer",
        )
        if "feature" not in merged.columns and "stability_feature" in merged.columns:
            merged["feature"] = merged["stability_feature"]
        if "feature" in merged.columns and "stability_feature" in merged.columns:
            mismatch = (
                merged["feature"].notna()
                & merged["stability_feature"].notna()
                & (merged["feature"].astype(str) != merged["stability_feature"].astype(str))
            )
            if bool(mismatch.any()):
                merged["feature_alt"] = merged["stability_feature"]
            merged = merged.drop(columns=["stability_feature"])
        order = [
            col for col in ("rank", "stability_rank", "selected_freq", "mean_rank", "score", "stability_score")
            if col in merged.columns
        ]
        if order:
            merged = merged.sort_values(
                order,
                ascending=[False if col in {"selected_freq", "score", "stability_score"} else True for col in order],
                na_position="last",
            ).reset_index(drop=True)
        return {
            "table": _feature_identity_index(merged),
            "graph": rank_graph,
            "label": rank_label,
        }

    if has_rank:
        return {
            "table": _feature_identity_index(rank_table.copy()),
            "graph": rank_cached.get("graph"),
            "label": rank_cached.get("label"),
        }

    return {
        "table": _feature_identity_index(_stability_feature_view(stab_table, rename_feature=False)),
        "graph": stab_cached.get("graph"),
        "label": stab_cached.get("label"),
    }

def _clear_sample_views(nt: "Netan") -> None:
    for key in ("aligned_samples_df", "label_maps"):
        nt._cache.pop(key, None)
    for key in ("samples", "nodes"):
        nt._results().pop(key, None)

def _clear_feature_views(nt: "Netan") -> None:
    nt._cache.pop("feature_meta", None)
    for key in ("features", "nodes"):
        nt._results().pop(key, None)

def _invalidate_graph_analysis_results(nt: "Netan") -> None:
    for key in ("scores", "best", "tune"):
        nt._results().pop(key, None)


def _cohere_feature_ranking_results(
    nt: "Netan",
    *,
    keep: str,
    graph_name: str,
    label: str,
) -> None:
    other = "rank_stability" if str(keep) == "rank" else "rank"
    cached = nt._results().get(other) or {}
    table = cached.get("table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        if cached.get("graph") != graph_name or cached.get("label") != str(label):
            nt._results().pop(other, None)
    nt._results().pop("shortlist", None)
    _clear_feature_views(nt)

def _drop_managed_feature_sync_cols(
    nt: "Netan",
    *,
    prefix: str = "netan",
) -> None:
    managed = _managed_feature_sync_columns(prefix=prefix)
    for rodin in nt.rodins:
        features_df = getattr(rodin, "features", None)
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            continue
        drop_cols = [col for col in map(str, features_df.columns) if col in managed]
        if drop_cols:
            setattr(rodin, "features", features_df.drop(columns=drop_cols))

def _auto_sync_rodins(
    nt: "Netan",
    *,
    sample_graphs: Optional[Sequence[str]] = None,
    sync_features: bool = False,
) -> None:
    if sample_graphs is not None:
        root = f"{_sync_prefix('netan')}_"
        for rodin in nt.rodins:
            samples_df = getattr(rodin, "samples", None)
            if not isinstance(samples_df, pd.DataFrame) or samples_df.empty:
                continue
            drop_cols = [
                col
                for col in map(str, samples_df.columns)
                if col.startswith(root) and col.endswith(("_community", "_module"))
            ]
            if drop_cols:
                setattr(rodin, "samples", samples_df.drop(columns=drop_cols))
        if sample_graphs:
            for graph_name in dict.fromkeys(map(str, sample_graphs)):
                _sync_samples_back_to_rodins(nt, graph=graph_name, prefix="netan", overwrite=True)
        _clear_sample_views(nt)
    if sync_features:
        _drop_managed_feature_sync_cols(nt)
        _sync_features_back_to_rodins(nt, prefix="netan", overwrite=True)
        _clear_feature_views(nt)

def _sync_samples_back_to_rodins(
    nt: "Netan",
    *,
    graph: Optional[str],
    prefix: str,
    overwrite: bool,
) -> None:
    graph_name, _, _ = nt._graph_context(graph)
    sync_cols = nt.samples(graph=graph_name)[["id", "community", "module"]].copy()
    sync_cols["id"] = sync_cols["id"].astype(str)
    graph_token = re.sub(r"[^0-9A-Za-z_]+", "_", str(graph_name)).strip("_").lower() or "value"
    root = f"{_sync_prefix(prefix)}_{graph_token}"
    sync_cols = sync_cols.rename(
        columns={
            "community": f"{root}_community",
            "module": f"{root}_module",
        }
    ).set_index("id")

    for rodin in nt.rodins:
        samples_df = _ensure_df(getattr(rodin, "samples", None), "r.samples").copy()
        sample_ids = _extract_sample_ids_from_df(samples_df, nt.sample_ids)
        if sample_ids is None:
            raise ValueError("Each rodin.samples table must contain unique sample IDs aligned to the current Netan object.")
        samples_df["_sample_id"] = list(map(str, sample_ids))
        join_cols = list(sync_cols.columns)
        if overwrite:
            samples_df = samples_df.drop(columns=[c for c in join_cols if c in samples_df.columns])
        else:
            join_cols = [c for c in join_cols if c not in samples_df.columns]
        if join_cols:
            samples_df = samples_df.join(sync_cols[join_cols], on="_sample_id")
        setattr(rodin, "samples", samples_df.drop(columns=["_sample_id"]))

def _sync_features_back_to_rodins(
    nt: "Netan",
    *,
    prefix: str,
    overwrite: bool,
) -> None:
    root = _sync_prefix(prefix)
    table = _feature_results_bundle(nt, require=False)["table"]

    for layer_name, rodin in zip(map(str, nt.names), nt.rodins):
        features_df = getattr(rodin, "features", None)
        if isinstance(features_df, pd.DataFrame) and not features_df.empty:
            base = _ensure_df(features_df, "r.features").copy()
            base.index = base.index.astype(str)
        else:
            X = _ensure_df(getattr(rodin, "X", None), "r.X")
            base = pd.DataFrame(index=X.index.astype(str))

        payload = None
        if isinstance(table, pd.DataFrame) and not table.empty:
            layer_rows = table[table["layer"].astype(str) == layer_name].copy()
            rename = {
                col: f"{root}_{col}"
                for col in _FEATURE_SYNC_COLUMNS
                if col in layer_rows.columns
            }
            if rename:
                part = layer_rows[["feature_id", *rename]].copy()
                part["feature_id"] = part["feature_id"].astype(str)
                payload = part.drop_duplicates("feature_id").set_index("feature_id").rename(columns=rename)

        if payload is None or payload.empty:
            setattr(rodin, "features", base)
            continue

        join_cols = list(payload.columns)
        if overwrite:
            drop_cols = [c for c in join_cols if c in base.columns]
            if drop_cols:
                base = base.drop(columns=drop_cols)
        else:
            join_cols = [c for c in join_cols if c not in base.columns]
        setattr(rodin, "features", base if not join_cols else base.join(payload[join_cols], how="left"))


def _format_public_param_parts(
    info: Dict[str, Any],
    semantics: Dict[str, str],
    *,
    nt: Optional["Netan"] = None,
    graph_name: Optional[str] = None,
    include_extras: bool = True,
    method_params: Optional[str] = None,
) -> List[str]:
    if semantics.get("kind") != "derived":
        parts = []
        if method_params is not None:
            parts.append(f"method_params={method_params or '-'}")
        if info.get("combine") is not None:
            parts.append(f"combine={info['combine']}")
        parts.extend(
            [
                f"thr_raw={_fmt_opt(info.get('thr_raw'))}",
                f"thr_norm={_fmt_opt(info.get('thr_norm'))}",
                f"auto={(info.get('auto_info') or {}).get('strategy') or info.get('auto') or '-'}",
                f"k={_display_k(info)}",
                f"mutual={bool(info.get('mutual', False))}",
            ]
        )
        if include_extras:
            parts.append(f"attach_isolates={bool(info.get('attach_isolates', False))}")
            if info.get("min_layers") is not None:
                parts.append(f"min_layers={int(info['min_layers'])}")
            if info.get("community_res") is not None:
                parts.append(f"community_res={float(info['community_res']):.4g}")
        return parts
    base = {
        "thr_raw_base": info.get("thr_raw_base"),
        "thr_norm_base": info.get("thr_norm_base"),
        "auto_base": info.get("auto_base"),
        "k_input_base": info.get("k_input_base"),
        "k_base": info.get("k_base"),
        "attach_isolates_base": info.get("attach_isolates_base"),
    }
    if not any(v is not None for v in base.values()):
        base = _derived_base_param_state(nt, str(graph_name)) if nt is not None and graph_name is not None else {}
    show_base = lambda value: value if isinstance(value, str) else _fmt_opt(value)
    parts = []
    if info.get("combine") is not None:
        parts.append(f"combine={info['combine']}")
    if method_params is not None:
        parts.append(f"method_params={method_params or '-'}")
    if base:
        parts.extend(
            [
                f"thr_raw[base]={show_base(base.get('thr_raw_base'))}",
                f"thr_norm[base]={show_base(base.get('thr_norm_base'))}",
                f"auto[base]={base.get('auto_base') or '-'}",
            ]
        )
        if base.get("k_input_base") is not None:
            parts.append(f"k_input[base]={base.get('k_input_base')}")
        parts.append(f"k[base]={base.get('k_base') or '-'}")
        if include_extras:
            parts.append(f"attach_isolates[base]={base.get('attach_isolates_base')}")
    if info.get("min_layers") is not None:
        parts.append(f"min_layers={int(info['min_layers'])}")
    if info.get("community_res") is not None:
        parts.append(f"community_res={float(info['community_res']):.4g}")
    return parts

def _mask_inactive_threshold_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(row.get("family") or "").lower()
    if family == "manual":
        row["auto_target"] = None
        row["auto"] = None
    elif family == "auto":
        row["thr_raw_input"] = None
        row["thr_norm_input"] = None
    return row

def _format_stats_parts(metrics: Dict[str, Any], *, failures: Optional[int] = None) -> List[str]:
    parts = [
        f"nodes={int(metrics.get('nodes', 0))}",
        f"edges={int(metrics.get('edges', 0))}",
        f"active_nodes={int(metrics.get('active_nodes', 0))}",
        f"mean_degree={float(metrics.get('mean_degree_active', 0.0)):.4g}",
        f"median_degree={float(metrics.get('median_degree_active', 0.0)):.4g}",
        f"max_degree={float(metrics.get('max_degree_active', 0.0)):.4g}",
        f"modules={int(metrics.get('modules', 0))}",
        f"communities={int(metrics.get('communities', 0))}",
        f"isolates={int(metrics.get('isolates', 0))}",
        f"density={float(metrics.get('density_active', 0.0)):.4g}",
    ]
    if failures is not None:
        parts.append(f"failures={int(failures)}")
    return parts

def _public_score_metrics(
    metrics: Dict[str, Any],
    objective: str,
    *,
    include_stats: bool = True,
) -> Dict[str, Any]:
    out = {key: metrics[key] for key in ("nodes", "edges", "active_nodes", "mean_degree_active", "median_degree_active", "max_degree_active", "modules", "communities", "isolates", "density_active") if include_stats and key in metrics}
    out.update({key: metrics[key] for key in ("score", "structure", "sep", "stab", "active_fraction") if key in metrics})
    out.update({label: metrics[src] for label, src in (("modularity", "modularity01"), ("degree_band", "degree_band"), ("module_size_band", "module_size_band")) if src in metrics})
    if objective == "supervised":
        out.update({key: metrics[key] for key in ("ari", "nmi", "label_assortativity") if key in metrics})
    out.update({key: metrics[key] for key in ("module_stability", "edge_stability") if key in metrics})
    return out

def _format_score_parts(metrics: Dict[str, Any], objective: str) -> List[str]:
    parts = [f"score={float(metrics.get('score', 0.0)):.4f}"]
    parts.extend(f"{label}={float(metrics.get(label, 0.0)):.4f}" for label in ("sep", "structure", "stab") if label in metrics)
    parts.extend(["\n", f"modularity={float(metrics.get('modularity01', 0.0)):.4f}", f"degree_band={float(metrics.get('degree_band', 0.0)):.4f}", f"module_size_band={float(metrics.get('module_size_band', 0.0)):.4f}"])
    if objective == "supervised":
        parts.extend(
            f"{label}={float(metrics.get(src, 0.0)):.4f}"
            for label, src in (
                ("ari", "ari"),
                ("nmi", "nmi"),
                ("label_assortativity", "label_assortativity"),
            )
        )
    parts.extend(f"{label}={float(metrics.get(label, 0.0)):.4f}" for label in ("module_stability", "edge_stability"))
    parts.extend(["\n", f"active_fraction={float(metrics.get('active_fraction', 0.0)):.4f}"])
    return parts

def _fmt_method_params(items: Dict[str, Any]) -> str:
    if not items:
        return "-"
    return ",".join(
        f"{key}={('None' if items[key] is None else (_fmt_opt(items[key]) if isinstance(items[key], float) else items[key]))}"
        for key in sorted(items)
    )

def _print_parts_block(label: str, parts: Sequence[str], *, indent: str = "  ", width: int = 108) -> None:
    parts = [str(part) for part in parts if str(part)]
    prefix = f"{indent}{label}: "
    pad = " " * len(prefix)
    line = prefix
    first_in_line = True

    def flush() -> None:
        nonlocal line, first_in_line
        if not first_in_line:
            print(line.rstrip())
        line = pad
        first_in_line = True

    for part in parts:
        if part == "\n":
            flush()
            continue
        token = str(part)
        sep = "" if first_in_line else " | "
        candidate = line + sep + token
        if not first_in_line and len(candidate) > int(width):
            print(line.rstrip())
            line = pad + token
            first_in_line = False
            continue
        line = candidate
        first_in_line = False
    flush()
