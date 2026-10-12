# ddmo v1.0.0

This is the first 1.0.0 release of the DDMO Python library for sparse
identification of ordinary differential equation systems.

## Included

- Candidate function libraries, strong and weak formulations, sparse solvers,
  and multi-objective identification through the `ddmo` Python API.
- Consistent piecewise-linear weak-form weights for the left- and right-hand
  sides, with regression tests for the discrete weak identity.
- An MIT license in the source and built distributions.

The frozen paper benchmark campaign and figures remain on the
`16-benchmark-paper` branch. They are not included in this release tree.

## Install

```bash
git clone --branch v1.0.0 https://gitlab.mpcdf.mpg.de/gducci/ddmo.git
cd ddmo
pip install .
```

The `ddmo` project on public PyPI is a different package. Install this release
from the GitLab repository.
