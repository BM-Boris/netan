# netan

[![PyPI version](https://img.shields.io/pypi/v/netan.svg)](https://pypi.org/project/netan/)
[![Python](https://img.shields.io/pypi/pyversions/netan.svg)](https://pypi.org/project/netan/)
[![License](https://img.shields.io/pypi/l/netan.svg)](LICENSE)

`netan` is a network analysis library for omics data. It builds sample or feature graphs from one or more matrices, tunes graph construction, ranks features against sample labels, and exports publication-ready or Cytoscape-ready outputs.

It works naturally with [rodin](https://github.com/BM-Boris/rodin) and Rodin-like objects.

Web app: [netan.io](https://netan.io)

## Highlights

- Build graphs in `samples` or `features` mode.
- Work with single-omics or multilayer multi-omics data.
- Use `spearman`, `clr`, `rf`, or `glasso` inference.
- Tune graph construction with a reusable `grid -> scores_grid -> materialize` workflow.
- Rank features with graph-aware label separation statistics.
- Visualize interactively with Plotly and export edge tables directly.

## Installation

```bash
pip install netan
```

Requires Python `>=3.10`.

## Quick start

```python
import rodin
import netan

r1 = rodin.create("metabolomics.csv", "meta.csv")
r2 = rodin.create("transcriptomics.csv", "meta.csv")

r1.transform()
r2.transform()

nt = netan.create([r1, r2])

nt.build(
    method="spearman",
    node_mode="samples",
    layer_mode="multilayer",
)

nt.plot(color="Group", title="Sample network")
```

## Core workflow

### Build

```python
nt.build(
    method="rf",
    node_mode="samples",
    layer_mode="multilayer",
    k="auto",
)
```

If you do not pass thresholds, `netan` sparsifies automatically through `auto_target`.

### Inspect

```python
nt.info()
nt.params()
nt.scores()
nt.edges()
```

### Tune

```python
nt.tune(label="Group")
nt.best()
```

For repeated score iteration on the same candidate space:

```python
nt.grid()

nt.scores_grid(
    label="Group",
    weights={
        "sep": [15, 5, 80],
        "supervised": [60, 15, 15, 10],
    },
)

nt.materialize()
```

### Rank and shortlist

```python
nt.rank("Group")
nt.stability_rank("Group")
nt_small = nt.shortlist(p_adj_max=0.01)
```

This workflow is available in `samples` mode and is useful when you want graph-aware feature selection before rebuilding a smaller network.

## Concepts

### Node mode

- `samples`: nodes are samples and edges represent sample similarity
- `features`: nodes are features and multilayer builds can include cross-omics feature edges

### Layer mode

- `stack`: one combined graph
- `multilayer`: per-layer graphs plus integrated outputs

### Methods

- `spearman`: correlation-based
- `clr`: mutual-information based
- `rf`: ExtraTrees similarity
- `glasso`: sparse precision graph

### Sparsity controls

- `auto_target`: default thresholding control
- `k=None`: threshold-only graph
- `k="auto"`: adaptive kNN pruning
- `mutual=True`: stricter kNN graph
- `attach_isolates=True`: reconnect isolates after sparsification

## Tuning model

`tune()` separates expensive candidate construction from cheap rescoring:

- `grid()`: build all candidate graph states once
- `scores_grid()`: score the grid in unsupervised or supervised mode
- `materialize()`: restore any leaderboard row as a live `Netan` object

This makes it practical to iterate on score weights without rerunning inference.

## Visualization and export

```python
nt.plot(layout="kamada_kawai", color="Group")
nt.export("edges.csv")
```

Supported layouts:

- `force-directed`
- `spring`
- `circular`
- `kamada_kawai`
- `random`

## Persistence

```python
path = nt.save("netan.pkl")
nt2 = netan.load(path)
```

This restores stored graphs, rankings, caches, and tuning grids. The live Plotly figure handle is not serialized.

## Input contract

`netan` expects one object or a list of objects exposing:

- `r.X`: `pandas.DataFrame` with shape `features x samples`
- `r.samples`: `pandas.DataFrame` whose first column contains sample IDs aligned to `r.X.columns`
- `r.features`: optional feature metadata

Rodin already matches this layout.

## Public API

Main entry points:

- `netan.create(...)`
- `netan.load(path)`
- `Netan.build(...)`
- `Netan.adjust(...)`
- `Netan.grid(...)`
- `Netan.scores_grid(...)`
- `Netan.materialize(...)`
- `Netan.tune(...)`
- `Netan.rank(...)`
- `Netan.stability_rank(...)`
- `Netan.shortlist(...)`
- `Netan.plot(...)`
- `Netan.export(...)`
- `Netan.save(path)`

For exact parameters, use the Python docstrings.

## Notes

- `samples` mode is where label-aware tuning and `rank()` make the most sense.
- `features` mode is the right choice for variable-level network exploration.
- For larger graphs, `spearman`, `clr`, and `rf` are usually the practical defaults.

## License

MIT
