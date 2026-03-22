# Netan — Multilayer Network Builder for Rodin‑like Objects

**Netan** builds multilayer networks from omics matrices and gives you clean APIs to analyze, visualize, export, and tune them. It supports Spearman, CLR (MI‑z), ExtraTrees‑RF, and Graphical Lasso; both *samples* and *features* node modes; stacked or multilayer graphs; optional raw/normalized thresholding; optional kNN sparsification; integrated outputs for all modes plus consensus outputs for multilayer sample networks; an interactive Plotly viewer; Cytoscape‑ready CSV export; and a two-stage `tune()` search that separates expensive inference from cheap graph adjustment.

Web App: https://netan.io

> **Works with any *Rodin‑like* object** exposing:
> - `r.X`: `pandas.DataFrame` (**features × samples**)
> - `r.samples`: `pandas.DataFrame` (first column = sample IDs; order matches `r.X.columns`)
> - `r.features` *(optional)*: `pandas.DataFrame` (index = feature IDs; used for tooltips/colors in *features* mode)
>
> See also: https://github.com/BM-Boris/rodin

---

## Installation

```bash
pip install netan
```

> Requires Python ≥ 3.10. Installs dependencies automatically: `rodin` (recent), `numpy`, `pandas`, `networkx`, `scikit-learn`, `joblib`, `tqdm`, `plotly`.

---

## Quick Start (Rodin-based)

Below is a **ready‑to‑run example** using two omics tables that share the same samples.

```python
import rodin
import netan

# 1) Create one or multiple Rodin objects from data + metadata
r1 = rodin.create("metabolomics.txt", "meta.csv")
r2 = rodin.create("transcriptomics.csv", "meta.csv")

# 2) Preprocess (Rodin handles normalization/log/scale etc.)
r1.transform()
r2.transform()

# 3) Build a multilayer network across shared samples
nt = netan.create([r1, r2])
nt.build(
    method="spearman",        # inference: 'spearman'|'clr'|'rf'|'glasso'
    thr_raw=0.75,             # threshold on abs(W_raw)
    layer_mode="multilayer",  # 'stack' or 'multilayer'
    node_mode="samples",      # 'samples' or 'features'
    k="auto",                 # threshold-density-based kNN restriction
    graph="entire",           # active graph: 'entire' | 'fused' | 'consensus' | layer name
)

# 4) Interactive Plotly graph (FigureWidget)
fig = nt.plot(
    title="Netan • Samples × Multilayer (Spearman, thr=0.75)",
    color="pGroup",           # column from r.samples to color nodes (optional)
    node_size=12,
    width=950,
    height=650,
)

# 5) Export an edge table compatible with Cytoscape
edges = nt.edges()
nt.export("edges.csv")
```

---

## Concepts at a Glance

- **Node mode**
  - `samples`: nodes are samples; edges reflect sample–sample similarity.
  - `features`: nodes are features; feature IDs are prefixed per input to avoid collisions.

- **Layer mode**
  - `stack`: combine all inputs into a single graph named `"entire"`; no consensus graph is created there.
  - `multilayer` in `samples` mode: keeps per-layer graphs in `G_layers`, a union graph in `G_all` (`entire`), a fused graph in `G_fused` (`fused`), and a consensus graph in `G_con`.
  - `multilayer` in `features` mode: keeps per-layer graphs in `G_layers`, a union graph in `G_all`, and a stored `cross` graph for cross-omics edges.

- **Graph semantics**
  - `fused` is a **direct** graph built from the fused matrix.
  - `cross` is a **direct** graph built from cross-omics feature pairs.
  - multilayer `entire` and `consensus` are **derived** graphs.
  - `adjust()` on a derived graph can cascade to the support graphs it depends on before rebuilding the derived graph.

- **Methods & thresholds**
  - Every method first computes a full `W_raw` matrix and a monotonic normalized `W_norm in [0,1]`.
  - Use `thr_raw` and/or `thr_norm`; if both are set, both must pass.
  - If neither threshold is given, Netan picks the strictest `thr_norm` found directly from `W_norm` such that at least `auto_target` of nodes still have at least one allowed edge before kNN. The default is `auto_target=0.95`, which prints as `auto=target95`.
  - If `thr_raw` or `thr_norm` is given, automatic thresholding is skipped and `auto_target` is ignored.
  - If `attach_isolates=True`, Netan reattaches isolated nodes with exactly one strongest edge when possible after thresholding and optional kNN, including manual-threshold builds.
  - Default `k="auto"` now inspects the active threshold graph: if it is already sparse, no kNN is applied; otherwise `k` grows gradually with active mean degree, is capped by typical neighborhood size, and never exceeds `10`.
  - Set `k=None` for threshold-only graphs, or `k=<int>` for threshold + kNN (union or mutual).

---

## Layouts (computed at plot time)

`plot()` computes node positions **after** applying UI filters (layer, `weight_min/weight_max`, `hide_isolated`). That means the layout reflects exactly what you visualize.

- Supported: `{ "force-directed", "spring", "circular", "kamada_kawai", "random" }`.
- `"force-directed"` is an alias for NetworkX `spring_layout`.
- Edge weights are passed to spring/force-directed, so stronger edges pull nodes closer.
- Use `layout_seed` for reproducibility in stochastic layouts.

---

## API Overview

### `create(rodins, names=None) -> Netan`
Builds a container from one or multiple Rodin‑like objects by aligning them to shared samples. Prints concise pre/post stats.

- **Parameters**
  - `rodins`: one object or a list of objects exposing `.X` and `.samples` (optionally `.features`, `.uns`).
  - `names`: optional list of human‑readable layer names (defaults to `L1`, `L2`, ...).

- **Returns**: `Netan` (with `.G` unset until you call `.build`).

---

### `Netan.build(method='spearman', node_mode='samples', layer_mode='stack', thr_raw=None, thr_norm=None, auto_target=0.95, attach_isolates=False, k='auto', mutual=False, min_layers=None, combine='mean', graph='entire', community_res=1.0, n_jobs=1, **kwargs) -> self`
Constructs integrated, per-layer, consensus, and active graphs as applicable to the selected mode.

- **Common parameters**
  - `method`: `'spearman' | 'clr' | 'rf' | 'glasso'`.
  - `node_mode`: `'samples' | 'features'`.
  - `layer_mode`: `'stack' | 'multilayer'`.
  - `thr_raw`: threshold on `abs(W_raw)`.
  - `thr_norm`: threshold on normalized similarity `W_norm in [0,1]`.
  - `auto_target`: target active-node fraction for automatic thresholding when no thresholds are supplied.
  - `attach_isolates`: reconnect isolated nodes with one strongest edge after thresholding and optional kNN.
  - `k`: kNN restriction. Default `"auto"` inspects the active threshold graph to decide whether kNN is needed and which `k` to use. Use `None` to disable it.
  - `mutual`: when `k` is set, use mutual-kNN instead of union-kNN.
  - `min_layers`: minimum supporting layers for `G_con` in multilayer samples mode.
  - `combine`: `'mean'|'median'|'max'` — fusion rule for multilayer aggregation.
  - `graph`: active graph for `plot()` / `edges()` / `export()`: `'entire' | 'fused' | 'consensus' | 'cross' | <layer name>`. `fused` and `consensus` are available only in multilayer `samples` mode; `cross` is available in multilayer `features` mode.
  - `community_res`: Louvain module resolution.
  - `n_jobs`: int — parallelism for CLR/RF computations.

- **Method‑specific `**kwargs`**
  - `clr`: `n_neighbors=int`.
  - `rf`: `n_estimators=int`, `max_depth=int|None` (0/''/None ⇒ `None`).
  - `glasso`: `alpha=float`, `max_iter=int`, `tol=float` (default `1e-4`).

- **Returns**: `self`. After the call:
  - `self.G_layers` stores per-layer graphs.
  - `self.G_all` stores the combined graph (`entire`).
  - `self.G_fused` stores the fused graph when available (`samples + multilayer`).
  - `self.G_con` stores the consensus graph when available (`None` in `stack` mode and in all `features` mode builds).
  - `self.G` points to the active graph selected by `graph=...`.
  - Nodes always carry active `community` / `module` labels. In multilayer builds, nodes also carry per-graph fields such as `community_entire`, `community_con`, `module_entire`, `module_con`, and per-layer variants.
  - In `features` mode, inter-layer edges are labeled as `cross`.

---

### `Netan.set_graph(graph='entire') -> self`
Switches the active graph used by `plot()`, `edges()`, and `export()` without rebuilding.

- `graph`: `'entire' | 'fused' | 'consensus' | 'cross' | <layer name>`; `fused` and `consensus` exist only in multilayer `samples` mode, `cross` in multilayer `features` mode.
- Updates the active graph selection used by `self.G`, `plot()`, `edges()`, and `export()`.
- Use `nt.graphs()` as a short alias for `nt.available_graphs()`.

---

### `Netan.info() -> None`
Prints the current summary view again without rebuilding.

- Uses the same per-graph summary format shown after `build()`.
- Reflects the current stored graphs after any `adjust()` calls.

---

### `Netan.params(graph=None) -> None`
Prints the effective parameters of a stored graph.

- If `graph=None`, uses the current active graph.
- Prints thresholds, auto mode, kNN, isolate attachment, consensus support, and Louvain resolution when applicable.
- Stores the full snapshot in `nt._results()["params"]`.

---

### `Netan.samples(graph=None) -> pandas.DataFrame`
Returns `r.samples` with graph-derived `community` and `module` columns.

- Available only in `samples` mode.
- If `graph=None`, uses the current active graph.
- Always returns a canonical `id` column aligned to the selected graph.
- Stores the result in `nt._results()["samples"]`.

---

### `Netan.nodes(graph=None, active_only=False) -> pandas.DataFrame`
Returns the node table for one graph variant.

- In `samples` mode, this is `samples(graph=...)` plus node-level graph attributes.
- In `features` mode, this is feature metadata plus node-level graph attributes and any currently available ranking columns.
- `active_only=True` removes degree-0 nodes after graph selection.

---

### `Netan.features() -> pandas.DataFrame`
Returns the feature table.

- Available in both modes, but mainly useful in `features` mode.
- Ranking columns are attached automatically when `rank()` and-or `stability_rank()` results exist.
- If both are available for the same `graph` and `label`, significance and stability columns are combined on `layer + feature_id`.
- Feature identity is carried canonically by `layer` + `feature_id`.

---

### `Netan.scores(graph=None, label=None) -> pandas.DataFrame`
Scores the current graph with the same graph-quality metrics used by `tune()`.

- If `graph=None`, uses the current active graph selected by `set_graph()`.
- If `label=None`, prints the unsupervised score block.
- If `label='<column>'` in `samples` mode, prints the supervised score block against `r.samples[<column>]`.
- Label values are treated as discrete classes. Numeric metadata columns are allowed, but they are not modeled as continuous targets.
- Stores the full result in `nt._results()["scores"]`.

---

### `Netan.rank(label, graph=None, layers=None, use_weights=True, standardize=True, n_perm=1000, seed=1, fdr=True, top=10, chunk_size=256) -> pandas.DataFrame`
Ranks features on a sample graph against a label column.

- Available only in `samples` mode.
- Label values are treated as discrete classes, even when the source metadata column is numeric.
- Uses a graph-aware multiclass pairwise class-contrast score with support-aware pair weighting.
- `p_perm` is the empirical permutation p-value.
- `p_adj` is Benjamini-Hochberg FDR.
- Stores the result in `nt._results()["rank"]`.

### `Netan.stability_rank(label, graph=None, layers=None, sample_frac=0.8, n_iter=50, top=20, stratify=True, use_weights=True, standardize=True, seed=1) -> pandas.DataFrame`
Estimates how stable the ranking is under repeated sample subsampling.

- Available only in `samples` mode.
- Reuses the same edge-based ranking statistic as `rank()`.
- Returns `selected_freq`, `mean_rank`, `median_rank`, `score`, and `score_sd`.
- Stores the result in `nt._results()["rank_stability"]`.

### `Netan.shortlist(n=None, layers=None, per_layer=False, p_adj_max=None, p_max=None, score_min=None, rank_max=None, selected_freq_min=None) -> Netan`
Returns a new `Netan` object restricted to selected ranked features.

- Uses the current automatic ranked feature view.
- If both `rank()` and `stability_rank()` are available for the same `graph` and `label`, filters can use both significance and stability columns together.
- With defaults, keeps every currently ranked feature from the latest automatic ranking view.
- `per_layer=True` applies `n` separately inside each layer.
- Layers with no surviving features are dropped from the new object.

### `Netan.best() -> pandas.DataFrame`
Returns the current best row from the latest `tune()` or `scores_grid()` run.

- Includes the winning graph, parameters, metrics, and final `score`.
- Reads from `nt._results()["tune"]` and stores the one-row table in `nt._results()["best"]`.

---

### `Netan.edges(graph=None) -> pandas.DataFrame`
Returns the edge table of a stored graph without writing a file.

- If `graph=None`, uses the current active graph.
- Columns include `source`, `target`, `weight`, `layer`, and `layers`.

---

### `Netan.export(path=None, graph=None, sep=',', index=False, float_format=None) -> pandas.DataFrame`
Writes a graph edge table to disk and returns it.

- If `graph=None`, uses the current active graph.
- If `path=None`, writes to `./netan_<graph>.csv`.
- Stores the last export target in `nt._results()["last_export"]`.

---

### `Netan.grid(node_mode=None, layer_modes=None, graphs=None, methods=None, method_grids=None, combine=None, auto_target=None, thr_norm=None, thr_raw=None, k=None, mutual=None, attach_isolates=None, min_layers=None, community_res=1.0, verbose=True, n_jobs=1) -> dict`
Builds the full tuning candidate grid once without scoring it.

- This is the expensive stage of tuning: it runs inference, builds candidate graph states, and caches the full grid bundle.
- The full grid is cached in `nt._cache["tune_grid"]`.
- The public summary table is cached in `nt._results()["grid"]["table"]`.
- Use `scores_grid()` afterwards to iterate on weights cheaply without rebuilding candidates.

### `Netan.scores_grid(grid=None, label=None, weights=None, top_results=10, apply=False, verbose=True) -> pandas.DataFrame`
Scores a prebuilt tuning grid and returns the ranked leaderboard.

- If `grid=None`, uses the latest cached grid from `grid()`.
- `label=None` scores the grid unsupervised; `label='<column>'` scores it supervised in `samples` mode.
- The leaderboard is cached in `nt._results()["scores_grid"]["table"]`.
- The latest winner payload is mirrored into `nt._results()["tune"]`, so `best()` always reads the current winner consistently.
- `materialize(candidate=0)` applies the first leaderboard row after a scoring run.

### `Netan.materialize(grid=None, candidate=None, apply=True) -> Netan`
Materializes one candidate from a tuning grid back into a live object.

- If `candidate=None`, applies the latest `scores_grid()` / `tune()` winner for that grid.
- If `candidate` is an integer, it is interpreted first as a zero-based row index into the latest `scores_grid()` leaderboard for that grid.
- To force exact `candidate_id` lookup, pass `{'candidate_id': <id>}`.
- With `apply=True`, updates the current object; with `apply=False`, returns a detached `Netan`.

---

### `Netan.adjust(graph=None, thr_raw=..., thr_norm=..., auto_target=..., attach_isolates=..., k=..., mutual=..., min_layers=..., combine=..., community_res=...) -> self`
Rebuilds a stored graph from cached matrices without re-running inference.

- If `graph=None`, adjusts the current active graph selected by `set_graph()`.
- Thresholds can be raised or lowered, `k` can be increased, decreased, or disabled, and isolate reattachment / mutual-kNN can be changed.
- Uses the matrices already stored in `self._cache['matrices']`; it does **not** call the inference methods again.
- Direct graphs (`entire` in `stack`, layers, `fused`, `cross`) are resparsified directly from their cached matrices.
- Derived graphs (`entire` / `consensus` in multilayer mode) are rebuilt from cached layer-level matrices and updated parameters.
- Only the selected graph is updated. Other stored graphs keep their existing structure, communities/modules, stats, and cached edge tables.

---

### `Netan.tune(node_mode=None, label=None, layer_modes=None, graphs=None, methods=None, method_grids=None, combine=None, auto_target=None, thr_norm=None, thr_raw=None, k=None, mutual=None, attach_isolates=None, min_layers=None, objective='auto', community_res=1.0, top_results=10, apply=True, verbose=True, n_jobs=1) -> pandas.DataFrame`
Searches for the strongest network configuration and optionally applies it to the current object.

- `tune()` separates:
  - **build search**: methods, method-specific parameters, `stack`/`multilayer`, and multilayer `combine`
  - **adjust search**: thresholds, `auto_target`, `k`, `mutual`, `attach_isolates`, and optional `min_layers`
- If `node_mode=None`, tuning uses the current object mode.
- Every stage-1 candidate is then expanded by local refine windows and rescored with the same final scoring formula used by `scores()`.
- With `apply=True`, the winning candidate is materialized directly from the cached tuning grid back into the current object.
- This preserves the full winning state (`G_layers`, `G_all`, `G_fused`, `G_con`, `G_cross`) rather than mutating only one graph in place.
- With `verbose=True`, tuning prints live progress such as `stage=1/2`, `iter=...`, `stage=2/2`, method parameters for each build iteration, and a final winner summary with score terms.

- **Default build search**
  - current object in `samples` mode
    - methods: `spearman`, `clr`, `rf`
    - `clr`: `n_neighbors=[2, 5]`
    - `rf`: `n_estimators=[160, 320]`, `max_depth=[None, 8]`
    - layer modes: `stack`, `multilayer`
    - target graphs: `entire`, `fused`
    - multilayer combine: `mean`
  - current object in `features` mode
    - methods: `spearman`, `clr`, `rf`
    - layer modes: `stack`, `multilayer`
    - target graph: `entire`
    - `cross` only if you ask for it explicitly
- `glasso` is never included by default; add it explicitly through `methods` / `method_grids`.

- **Default adjust search**
  - auto family: `auto_target=[0.90, 0.94, 0.98]`
  - manual threshold families: off by default (`thr_norm=[]`, `thr_raw=[]`)
  - sparsity: `k=[None, 'auto']`
  - mutual kNN: `mutual=[False]`
  - isolate reattachment: `attach_isolates=[False]`
  - `min_layers` is off by default

- **Scoring**
  - `label=None` → unsupervised final score combining graph structure, local stability, and active-node coverage
  - `label='<column>'` in `samples` mode → supervised final score combining label separation, graph structure, local stability, and active-node coverage
  - supervised label values are treated as discrete classes, even when the source metadata column is numeric
  - supervised tuning is not available in `features` mode

- **Returns**
  - `tune()` returns the public leaderboard as a `pandas.DataFrame`
  - the leaderboard is also cached in `nt._results()["scores_grid"]["table"]`
  - the mirrored winner payload is stored in `nt._results()["tune"]`
  - the full candidate grid bundle remains cached in `nt._cache["tune_grid"]`
  - after `tune(apply=False)`, call `nt.materialize()` or `nt.materialize(candidate=0)` to apply a leaderboard row later

Example:

```python
nt.tune(
    label="pGroup",
    methods=["spearman", "rf"],
    combine=["mean", "max", "median"],
    method_grids={"rf": {"n_estimators": [160, 320], "max_depth": [None, 8]}},
    auto_target=[0.95, 0.98],
    k=[None, "auto"],
    attach_isolates=[False, True],
)

nt._results()["tune"]["best_build_params"]
nt._results()["tune"]["best_adjust_params"]
nt._results()["tune"]["table"].head()
```

---

### `Netan.plot(graph=None, color=None, size=None, shape=None, layer=None, hide_isolated=False, weight_min=None, weight_max=None, node_size=10, width=None, height=None, title=None, continuous_colorscale='Viridis', layout='force-directed', layout_seed=777) -> plotly.graph_objs.FigureWidget`
Creates an interactive Plotly network.

- **Color/size/shape**
  - *Categorical* color/shape: nodes split into legend groups; toggling legend hides incident edges live.
  - `size` expects a continuous numeric node column.
  - *Continuous* color: shows a colorbar; if `shape=None`, categorical legend toggles are disabled.
  - If `color`, `size`, and `shape` are all omitted, the legend is hidden.

- **Graph**
  - `graph`: render another stored graph without changing the active graph on the object.
  - in `features` mode, plotting can use any ranking columns currently exposed by `features()`.

- **Layer/weight filters**
  - `layer`: keep an edge if this label is present in its `layers` set.
  - `weight_min/max`: numeric bounds to prune edges.
  - `hide_isolated`: optionally drop nodes with no edges after filtering.

- **Layout**
  - `{ 'force-directed','spring','circular','kamada_kawai','random' }`; set `layout_seed` for reproducibility.

- **Returns**: a `FigureWidget` suitable for notebooks/dashboards.

---

### `Netan.to_csv(path=None, graph=None, sep=',', index=False, float_format=None) -> pandas.DataFrame`
Alias for `edges()` / `export()`.

- If `path=None`, returns the table without writing a file.
- If `path` is provided, writes the table and returns it.

---

### `Netan.save(path) -> str`
Pickles the current `Netan` object to disk.

- Saves graphs, caches, tuning grids, rankings, and stored results.
- The interactive `fig` handle is omitted from the pickle; after reload it is `None`.

### `netan.load(path) -> Netan`
Restores a pickled `Netan` object from disk.

- Intended for local workflow checkpoints in a compatible Python environment.

---

## Threshold Tips

- **Spearman / Glasso**: start with `thr_raw=0.6–0.9`.
- **CLR**: start with `thr_raw≈2–5` or use `thr_norm`.
- **RF (ExtraTrees)**: start with `thr_raw≈0.02–0.10` or use `thr_norm`.
- **Unknown scale / mixed methods**: prefer `thr_norm` or leave thresholds unset and let Netan pick one automatically from `auto_target`.
- **Need a pure threshold graph**: set `k=None`.
- **Need a stricter sparse graph**: increase `thr_raw` / `thr_norm`, lower `k`, or enable `mutual=True`.
- **Need to keep manual thresholds exact**: pass `thr_raw` and/or `thr_norm` and leave `attach_isolates=False`.

---

## Performance & Limits

- Soft **density guard** around ~10,000 edges (`MAX_EDGES`): warnings suggest raising thresholds or reducing variables.
- Complexity (roughly):
  - `spearman/CLR/RF` ~ O(p²) in the number of nodes per layer.
  - `glasso` ~ O(p³); consider increasing `alpha` or reducing dimensionality.
- Use `n_jobs` to parallelize CLR/RF.

---

## Troubleshooting

- **Graph too dense** → raise `thr_raw` / `thr_norm`, add `k`, switch to a stricter method (`glasso`), or reduce variables.
- **`GraphicalLasso failed`** → increase `alpha` (e.g., `0.1–0.2`), relax `tol`, ensure scaling is appropriate.
- **Empty plot** → check `layer`/`weight_min/max` filters and that inputs share sample IDs.
- **Too many categories for `shape`** → map values to fewer categories (limited symbol set).

---

## License

MIT (see `LICENSE`).
