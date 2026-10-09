# Reproduce one Figure 2 run

Figure 2 summarizes **ten** Goodwin identification runs. This tutorial reproduces
one of them: initial-condition block 4. It produces one selected coefficient
matrix and one test score; the selection-frequency heatmap requires all ten
runs. The executable template is
[`examples/quickstart/goodwin_joint.py`](../../examples/quickstart/goodwin_joint.py).

The bundled block 4 reference uses the corrected weak-form weights. The old
ten-run artwork has been removed; its coordinated replacement awaits the SI
figure review. Earlier numerical campaign data remain archived for audit.

From the repository root, install the versions used for the saved campaign:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . "numpy==2.3.1" "scipy==1.16.0" "scikit-learn==1.7.0" "pysindy==1.7.5" "pymoo==0.6.1.5" "setuptools<81"
```

First check the workflow with 36 evaluations and a shorter trajectory:

```bash
python examples/quickstart/goodwin_joint.py --smoke
```

Then run the full Figure 2 case:

```bash
python examples/quickstart/goodwin_joint.py --block 4
```

The full command evaluates **2,400** candidates and can take several minutes.
It asserts that the selected Pareto index, coefficients, validation score, and
test score agree with the corrected-weight block 4 reference in
`benchmarks/data/figure2_tutorial_block04.json`. The default
block 4 should select Pareto model **7**, with validation NRMSE **0.030507759**
and test NRMSE **0.033232010** under the pinned environment.

## Workflow

1. The script reads block 4's Latin-hypercube initial conditions and generates
   all six designed trajectories on the $0$–$200$ time grid. It then drops the
   unused third training trajectory, retaining two training, one validation,
   and two test trajectories. Generating all six preserves the original noise
   stream. Noise is 10% of each state's standard deviation.
2. Savitzky–Golay preprocessing smooths the training observations. The weak
   formulation fits integrated equations without supplying pointwise
   derivatives.
3. The candidate library includes degree-2 polynomial terms, Hill inhibition
   with an optimized exponent, and an exponential term with an optimized rate.
   NSGA-II also searches weak-window and STLSQ settings using trajectory RMSE,
   Durbin–Watson deviation, and active-coefficient count.
4. The model with the lowest **clean validation** rollout NRMSE is selected.
   Ties prefer fewer active coefficients, then lower training RMSE. Clean
   validation data are available here because this is a synthetic experiment;
   replace this rule with a suitable observed-data metric for real data.
5. The two test trajectories are used only after selection. The script prints
   the selected equations and checks the result against the frozen campaign.

Block 4 in the corrected campaign selects an exponential feedback term in the
$\dot x$ equation. A good trajectory score does not imply recovery of the true
Hill term. This is one example of the library-selection ambiguity summarized
across all ten runs in Figure 2.

To reproduce another block, pass `--block 1` through `--block 10` and
`--saved-run` with the corresponding corrected result JSON. The full ten-run
campaign will be frozen with the revised supplementary figures. For a new
system, replace the synthetic data-generation section and
initial conditions, then adjust the candidate library, bounds, and validation
rule in the script.
