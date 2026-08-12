# Compatibility paths

Postprocessing is now organized by manuscript figure under
`benchmarks/figures/`. The previous Python entry point remains available for
existing commands, but new work should use:

```bash
python -m benchmarks.figures.figure_1.postprocess
```
