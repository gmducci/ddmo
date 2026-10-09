# Benchmark data

- [`current/`](current/) contains the frozen corrected Figure 1 and Figure 2
  campaigns: configurations, provenance, selected-run JSON, saved trajectories,
  a clean-run verification, and checksums. Use `python -m benchmarks.freeze`
  from the repository root to validate the snapshot.
- [`figure2_tutorial_block04.json`](figure2_tutorial_block04.json) is the
  compact corrected-weight reference for the Figure 2 tutorial.
- [`historical/`](historical/) contains the earlier campaigns and checksum
  manifest. The weak-form runs used the previous weights and are retained
  solely for audit.

The selected paper artwork is in [`../figures/`](../figures/).
