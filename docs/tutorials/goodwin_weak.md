# Run one Figure 1 weak-form design

This tutorial runs **one** of the Figure 1 “Weak, optimized” designs. Choose one
noise level and one seed; the script generates the same Goodwin trajectories,
fits the same candidate library, and uses the same optimization settings and
model-selection rule. It uses the corrected weak-form weights. The saved
Figure 1 campaign in `benchmarks/data/historical/` used the earlier weights,
so its selected model and scores can differ.

For the campaign environment, use Python **3.12.4** and install the
recorded dependency versions in a fresh environment. From the
repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . "numpy==2.3.1" "scipy==1.16.0" "scikit-learn==1.7.0" "pysindy==1.7.5" "pymoo==0.6.1.5" "setuptools<81"
```

Then run:

```bash
python examples/quickstart/goodwin_weak.py
```

The defaults are **10% noise** and **seed 11**. To reproduce another point:

```bash
python examples/quickstart/goodwin_weak.py --noise 0.20 --seed 55
```

Noise can be `0`, `0.05`, `0.10`, `0.15`, or `0.20`. Seed can be `11`, `22`,
`33`, `44`, or `55`. These are the 25 “Weak, optimized” cases in Figure 1.
Each command runs just the chosen case. A full run makes **2,400 candidate
evaluations**, so it can take several minutes.

## What the script does

1. It generates six Goodwin trajectories on the benchmark time grid, using the
   chosen seed for the observation noise.
2. It fits on three noisy trajectories. The candidate library contains degree-2
   polynomials, interactions, and a Hill term with exponent 9.5.
3. NSGA-II searches the weak-window settings and STLSQ settings with the chosen
   seed. Its three objectives are training trajectory RMSE, Durbin–Watson
   deviation, and the number of active coefficients.
4. It simulates each valid Pareto model on the **one validation trajectory**.
   The model with the lowest validation rollout NRMSE is selected. Ties prefer
   fewer active coefficients, then lower training trajectory RMSE.
5. It tests that selected model on the **two held-out trajectories** and prints
   all three Figure 1 metrics: test NRMSE, coefficient relative error, and
   false-positive coefficient count. It also prints the historical saved
   values and the difference between the selected coefficient matrices.

The historical plotted Figure 1 point was the median of five selected test
results at that noise level. This script runs one design, not the median.

In the corrected local campaign, the default design selected Pareto model 12
with test NRMSE **0.14967033**. The historical saved run selected model 0 with
test NRMSE **0.16668603**. The numerical comparison in the script makes this
change visible; it is not an assertion that the two weight calculations agree.
