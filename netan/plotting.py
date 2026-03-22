"""Plotting helpers for Netan interactive graph rendering."""

import os
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import FigureWidget


_LAYOUT_FUNCS = {
    "force-directed": nx.spring_layout,
    "spring": nx.spring_layout,
    "circular": nx.circular_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "random": nx.random_layout,
}


def _ensure_df(x, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas.DataFrame")
    return x


def _compute_layout(G: nx.Graph, layout: str = "force-directed", seed: int = 777, weighted: bool = True):
    algo = "force-directed" if layout == "force-directed" else layout
    func = _LAYOUT_FUNCS.get(algo)
    if func is None:
        raise ValueError(
            f"Unknown layout '{layout}'. Allowed layouts: {sorted(_LAYOUT_FUNCS)}"
        )
    kw = {}
    if algo in ("force-directed", "spring"):
        kw["seed"] = seed
    if weighted and algo in ("force-directed", "spring", "kamada_kawai"):
        kw["weight"] = "weight"
    return func(G, **kw)


def plot_netan(
    model: Any,
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
    layout_seed: int = 777,
) -> FigureWidget:
    """
    Build a Plotly `FigureWidget` from one graph on a Netan-like model.

    The wrapper `Netan.plot()` is the public entry point. This helper assumes
    the model can resolve the selected graph and applies optional layer,
    weight, and isolate filters before computing the layout and traces. The
    created figure is written back to ``model.fig`` before return.
    """
    color_arg = color
    shape_arg = shape
    graph_choice = getattr(model, "_active_graph_name", lambda: "entire")() if graph in (None, "") else str(graph)
    if not hasattr(model, "_resolve_active_graph"):
        if model.G is None:
            raise RuntimeError("Build the network first (.build).")
        graph_name, G = str(graph_choice), model.G
    else:
        graph_name, G = model._resolve_active_graph(graph_choice)
    if G is None:
        raise RuntimeError("Build the network first (.build).")

    if not isinstance(node_size, int) or node_size <= 0:
        raise ValueError("node_size must be a positive integer.")
    if width is not None and (not isinstance(width, int) or width <= 0):
        raise ValueError("width must be a positive integer.")
    if height is not None and (not isinstance(height, int) or height <= 0):
        raise ValueError("height must be a positive integer.")

    def _is_number(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    def _col_type(values: pd.Series) -> str:
        vals = [v for v in values if v is not None and v == v]
        if not vals:
            return "categorical"
        if all(_is_number(v) for v in vals) and len(set(map(float, vals))) >= 6:
            return "continuous"
        return "categorical"

    def _cat_key(value):
        return (True, "") if pd.isna(value) else (False, str(value))

    def _cat_label(key) -> str:
        return "(missing)" if key[0] else key[1]

    def _display_label(row: pd.Series) -> str:
        for key in ("feature_id", "id"):
            value = row.get(key)
            if pd.notna(value) and str(value) not in {"", "None", "nan"}:
                return str(value)
        return str(row.get("id", ""))

    node_mode = (model._meta or {}).get("nodeMode", "samples")
    nodes_df = _ensure_df(model.nodes(graph=graph_name), "nodes()").copy()
    nodes_df = nodes_df.set_index("id", drop=False)
    if not color and node_mode == "features" and "layer" in nodes_df.columns:
        color = "layer"

    available_cols = set(nodes_df.columns)
    rank_hint = (
        " Available ranked metadata appears automatically after rank() and/or stability_rank()."
        if node_mode == "features"
        else ""
    )

    def _require_metadata_col(name: Optional[str]) -> None:
        if name and name not in available_cols:
            raise ValueError(
                f"Column '{name}' not found in node metadata. "
                f"Available: {sorted(available_cols)}.{rank_hint}"
            )

    for col in (color, size, shape):
        _require_metadata_col(col)

    edges_df = _ensure_df(model.edges(graph=graph_name), "edges()").copy()
    if "layers" not in edges_df.columns:
        edges_df["layers"] = ""
    def _layer_tokens(value) -> set[str]:
        if isinstance(value, (set, list, tuple)):
            return {str(v) for v in value if str(v)}
        if isinstance(value, str):
            return {token for token in value.split("|") if token}
        if value is None or pd.isna(value):
            return set()
        return {str(value)}

    edges_df["layers"] = edges_df["layers"].map(_layer_tokens)
    edge_free = edges_df.empty

    if layer:
        allowed_layers = sorted({token for vals in edges_df["layers"] for token in vals if token})
        if layer not in allowed_layers:
            raise ValueError(f"Unknown layer '{layer}'. Allowed layers: {allowed_layers}")

    if not edge_free:
        wmin = float(edges_df["weight"].min())
        wmax = float(edges_df["weight"].max())
        if weight_min is None:
            weight_min = wmin
        if weight_max is None:
            weight_max = wmax
        if weight_min > weight_max:
            raise ValueError("weight_min cannot be greater than weight_max.")
        if weight_min < wmin or weight_max > wmax:
            raise ValueError(f"weight_min/max must be within [{wmin:.6g}, {wmax:.6g}].")

        if layer:
            edges_df = edges_df[edges_df["layers"].apply(lambda s: layer in s)].copy()

        edges_df = edges_df[
            (edges_df["weight"] >= float(weight_min)) &
            (edges_df["weight"] <= float(weight_max))
        ].copy()
    else:
        weight_min = weight_max = None

    if hide_isolated:
        keep_ids = set(edges_df["source"]).union(set(edges_df["target"]))
        nodes_df = nodes_df[nodes_df["id"].isin(keep_ids)].copy()

    H = nx.Graph()
    H.add_nodes_from(nodes_df["id"].astype(str).tolist())
    for _, r in edges_df.iterrows():
        s = str(r["source"])
        t = str(r["target"])
        w = float(r["weight"])
        H.add_edge(s, t, weight=w)

    pos_raw = _compute_layout(H, layout=layout, seed=layout_seed, weighted=True)

    def _xy(i):
        i = str(i)
        if i in pos_raw:
            x, y = pos_raw[i]
            return float(x), float(y)
        return (float(np.random.uniform(-1, 1)), float(np.random.uniform(-1, 1)))

    xy = nodes_df["id"].map(_xy)
    nodes_df["x"] = [t[0] for t in xy]
    nodes_df["y"] = [t[1] for t in xy]

    for _, r in nodes_df.iterrows():
        nid = str(r["id"])
        if nid in G:
            G.nodes[nid]["x"] = float(r["x"])
            G.nodes[nid]["y"] = float(r["y"])

    pos = nodes_df.set_index("id")[["x", "y"]].astype(float).to_dict(orient="index")

    def build_edge_xy(visible_node_ids: set[str]):
        ex, ey = [], []
        for _, r in edges_df.iterrows():
            s, t = r["source"], r["target"]
            if s in visible_node_ids and t in visible_node_ids:
                ps, pt = pos.get(s), pos.get(t)
                if ps and pt:
                    ex += [ps["x"], pt["x"], None]
                    ey += [ps["y"], pt["y"], None]
        return ex, ey

    initial_visible_ids = set(nodes_df["id"].astype(str).tolist())
    ex0, ey0 = build_edge_xy(initial_visible_ids)

    if edge_free:
        mean_width = 1.0
    else:
        denom = max(wmax - wmin, 1e-12)
        widths = 0.75 + 3.0 * ((edges_df["weight"] - wmin) / denom)
        mean_width = float(widths.mean()) if len(widths) else 1.0

    base_edge_trace = go.Scatter(
        x=ex0,
        y=ey0,
        mode="lines",
        line=dict(color="#888888", width=mean_width),
        hoverinfo="none",
        showlegend=False,
        name="edges",
    )
    highlight_edge_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line=dict(color="#444444", width=mean_width * 1.4),
        hoverinfo="none",
        showlegend=False,
        name="highlight_edges",
    )

    traces = [base_edge_trace, highlight_edge_trace]

    def _series(name: Optional[str]) -> pd.Series:
        if not name or name not in nodes_df.columns:
            return pd.Series([None] * len(nodes_df), index=nodes_df.index)
        return nodes_df[name]

    color_s = _series(color)
    size_s = _series(size)
    shape_s = _series(shape)
    color_kind = _col_type(color_s.dropna()) if color else "categorical"
    shape_keys = shape_s.map(_cat_key)
    show_legend = bool(shape_arg) or bool(color_arg and color_kind != "continuous")

    if size:
        size_num = pd.to_numeric(size_s, errors="coerce")
        if size_num.notna().sum() == 0 or size_num.nunique(dropna=True) < 2:
            raise ValueError(f"Column '{size}' must be continuous numeric metadata for size scaling.")
        smin = float(size_num.min())
        smax = float(size_num.max())
        if np.isclose(smin, smax):
            size_scaled = pd.Series(float(node_size), index=nodes_df.index)
        else:
            size_scaled = 8.0 + 18.0 * ((size_num - smin) / max(smax - smin, 1e-12))
            size_scaled = size_scaled.fillna(float(node_size))
    else:
        size_scaled = pd.Series(float(node_size), index=nodes_df.index)

    symbols = ["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star"]

    if shape:
        n_shapes = len(set(shape_keys.tolist()))
        if n_shapes > len(symbols):
            raise ValueError(
                f"'shape' column '{shape}' has {n_shapes} unique values; "
                f"max symbols = {len(symbols)}."
            )

    if color and color_kind == "continuous":
        if pd.to_numeric(nodes_df[color], errors="coerce").notna().sum() == 0:
            raise ValueError(f"Column '{color}' must be numeric for continuous coloring.")
        if shape and _col_type(shape_s.dropna()) == "continuous":
            raise ValueError("'shape' should be categorical.")

        shp_vals = sorted(set(shape_keys.tolist()))
        shp_map = {key: symbols[i % len(symbols)] for i, key in enumerate(shp_vals)}
        for i, s_key in enumerate(shp_vals):
            s_label = _cat_label(s_key)
            sub = nodes_df[shape_keys.map(lambda key: key == s_key)]
            hover = [
                f"ID: {_display_label(r)}"
                + (f"<br>{color}: {r[color]}" if color else "")
                + (f"<br>{size}: {r[size]}" if size else "")
                + (f"<br>{shape}: {s_label}" if shape else "")
                for _, r in sub.iterrows()
            ]
            traces.append(
                go.Scatter(
                    x=sub["x"].astype(float),
                    y=sub["y"].astype(float),
                    mode="markers",
                    name=("nodes" if not shape else str(s_label)),
                    hoverinfo="text",
                    text=hover,
                    showlegend=show_legend,
                    customdata=sub["id"].astype(str),
                    marker=dict(
                        color=pd.to_numeric(sub[color], errors="coerce"),
                        colorscale=continuous_colorscale,
                        showscale=(i == 0),
                        colorbar=(dict(title=color) if i == 0 else None),
                        symbol=shp_map.get(s_key, "circle"),
                        size=size_scaled.loc[sub.index].astype(float),
                        opacity=1.0,
                        line=dict(width=1, color="#333333"),
                    ),
                )
            )
    else:
        palette = [
            "#c0392b",
            "#2980b9",
            "#27ae60",
            "#e67e22",
            "#8e44ad",
            "#8d6e63",
            "#d81b60",
            "#7f8c8d",
            "#f4c20d",
            "#00acc1",
            "#ad1457",
            "#afc52f",
            "#556b2f",
            "#6d214f",
            "#303f9f",
            "#bdc3c7",
            "#9b59b6",
            "#3f51b5",
            "#ff7043",
            "#c0b283",
            "#40e0d0",
        ]
        color_keys = color_s.map(_cat_key)
        color_vals = sorted(set(color_keys.tolist()))
        shape_vals = sorted(set(shape_keys.tolist()))
        c_map = {key: palette[i % len(palette)] for i, key in enumerate(color_vals)}
        s_map = {key: symbols[i % len(symbols)] for i, key in enumerate(shape_vals)}

        nodes_df["_color_key"] = list(color_keys)
        nodes_df["_shape_key"] = list(shape_keys)

        for (c_key, s_key), sub in nodes_df.groupby(["_color_key", "_shape_key"], sort=False):
            c_val = _cat_label(c_key)
            s_val = _cat_label(s_key)
            name = (
                c_val
                if (color and not shape)
                else (s_val if (shape and not color) else ("nodes" if not color and not shape else f"{c_val} | {s_val}"))
            )
            hover = [
                f"ID: {_display_label(r)}"
                + (f"<br>{color}: {c_val}" if color else "")
                + (f"<br>{size}: {r[size]}" if size else "")
                + (f"<br>{shape}: {s_val}" if shape else "")
                + (f"<br>compound: {r.get('compound', '')}" if "compound" in r and r.get("compound") else "")
                for _, r in sub.iterrows()
            ]
            traces.append(
                go.Scatter(
                    x=sub["x"].astype(float),
                    y=sub["y"].astype(float),
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    hoverinfo="text",
                    text=hover,
                    showlegend=show_legend,
                    customdata=sub["id"].astype(str),
                    marker=dict(
                        color=c_map.get(c_key, "#000000"),
                        symbol=s_map.get(s_key, "circle"),
                        size=size_scaled.loc[sub.index].astype(float),
                        opacity=1.0,
                        line=dict(width=1, color="#333333"),
                    ),
                )
            )
        nodes_df.drop(columns=["_color_key", "_shape_key"], inplace=True)

    base_edge_idx = 0
    highlight_edge_idx = 1
    node_trace_indices = list(range(2, len(traces)))

    fig_layout = go.Layout(
        title=title,
        hovermode="closest",
        showlegend=show_legend,
        margin=dict(l=20, r=60, t=50, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=width,
        height=height,
        uirevision="netan",
    )
    fig = FigureWidget(data=traces, layout=fig_layout)

    highlight_node_trace = go.Scatter(
        x=[],
        y=[],
        mode="markers",
        marker=dict(
            size=node_size + 2,
            symbol="circle-open",
            color="#444444",
            opacity=1.0,
            line=dict(width=3),
        ),
        hoverinfo="none",
        showlegend=False,
        name="highlight_nodes",
    )
    fig.add_trace(highlight_node_trace)
    highlight_node_idx = len(fig.data) - 1

    pinned_ids: set[str] = set()
    highlight_centers: set[str] = set()

    def _visible_node_ids() -> set[str]:
        ids: set[str] = set()
        for idx in node_trace_indices:
            tr = fig.data[idx]
            vis = tr.visible
            if vis in (True, None):
                cd = getattr(tr, "customdata", None)
                if cd is not None:
                    ids.update(map(str, cd))
        return ids

    def _rebuild_base_edges():
        visible_ids = _visible_node_ids()
        ex, ey = build_edge_xy(visible_ids)
        with fig.batch_update():
            fig.data[base_edge_idx].x = ex
            fig.data[base_edge_idx].y = ey

    def _update_pinned_labels():
        visible = _visible_node_ids()
        annotations = []

        for nid in pinned_ids:
            nid = str(nid)
            if nid not in visible:
                continue
            p = pos.get(nid)
            if p is None:
                continue

            row = nodes_df.loc[nid]
            compound_val = row.get("compound", None)
            if isinstance(compound_val, str) and compound_val.strip():
                label_text = compound_val.strip()
            else:
                label_text = _display_label(row) or nid

            annotations.append(
                dict(
                    x=p["x"],
                    y=p["y"],
                    text=f"<b>{label_text}</b>",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    yshift=8,
                    font=dict(size=12),
                )
            )

        with fig.batch_update():
            fig.layout.annotations = tuple(annotations)

    def _clear_pinned_labels():
        pinned_ids.clear()
        with fig.batch_update():
            fig.layout.annotations = ()

    def _clear_highlight():
        highlight_centers.clear()
        with fig.batch_update():
            fig.data[highlight_edge_idx].x = []
            fig.data[highlight_edge_idx].y = []
            fig.data[highlight_node_idx].x = []
            fig.data[highlight_node_idx].y = []

    def _full_reset():
        _clear_highlight()
        _clear_pinned_labels()

    def _update_highlight():
        visible = _visible_node_ids()
        centers = {c for c in highlight_centers if c in visible}
        if not centers:
            _clear_highlight()
            return

        neigh = set(centers)
        hex_coords, hey_coords = [], []

        for _, r in edges_df.iterrows():
            s = str(r["source"])
            t = str(r["target"])
            if ((s in centers) or (t in centers)) and (s in visible) and (t in visible):
                neigh.add(s)
                neigh.add(t)
                ps = pos.get(s)
                pt = pos.get(t)
                if ps and pt:
                    hex_coords += [ps["x"], pt["x"], None]
                    hey_coords += [ps["y"], pt["y"], None]

        hnx, hny = [], []
        for nid in neigh:
            nid = str(nid)
            if nid not in visible:
                continue
            p = pos.get(nid)
            if p:
                hnx.append(p["x"])
                hny.append(p["y"])

        with fig.batch_update():
            fig.data[highlight_edge_idx].x = hex_coords
            fig.data[highlight_edge_idx].y = hey_coords
            fig.data[highlight_node_idx].x = hnx
            fig.data[highlight_node_idx].y = hny

    def _handle_node_click(trace, points, _state):
        if not points.point_inds:
            return
        idx = int(points.point_inds[0])
        cd = getattr(trace, "customdata", None)
        if cd is None or idx >= len(cd):
            return
        nid = str(cd[idx])

        if nid not in pinned_ids:
            pinned_ids.add(nid)
            _update_pinned_labels()
            return

        if nid in highlight_centers:
            highlight_centers.remove(nid)
        else:
            highlight_centers.add(nid)
        _update_highlight()

    def _on_visible_change(*_args):
        _rebuild_base_edges()
        _update_pinned_labels()
        _update_highlight()

    _reset_state = {"zoomed": False, "busy": False}

    def _on_xrange_change(layout, _new_range):
        if layout.xaxis.autorange is False:
            _reset_state["zoomed"] = True

    def _on_xautorange_change(layout, new_value):
        if _reset_state["busy"]:
            return
        if new_value is True:
            _reset_state["busy"] = True
            try:
                if _reset_state["zoomed"]:
                    _reset_state["zoomed"] = False
                else:
                    _full_reset()
                if layout.xaxis.autorange is True:
                    layout.xaxis.autorange = False
            finally:
                _reset_state["busy"] = False

    fig.layout.on_change(_on_xrange_change, "xaxis.range")
    fig.layout.on_change(_on_xautorange_change, "xaxis.autorange")

    for idx in node_trace_indices:
        tr = fig.data[idx]
        tr.on_change(_on_visible_change, "visible")
        try:
            tr.on_click(_handle_node_click)
        except Exception:
            pass

    model.fig = fig

    if ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_GPU" in os.environ):
        try:
            from IPython.display import display

            display(fig)
        except Exception:
            pass

    return fig
