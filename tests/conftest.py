"""Use a noninteractive plotting backend for local and CI test runs."""

import matplotlib

matplotlib.use("Agg")
