# src/ddmo/__init__.py

from pathlib import Path

from ddmo.formulation.formulation_settings import FormulationSettings
from ddmo.main import identify
from ddmo.solver.settings import SolverSettings
from ddmo.utils.constructors import IdentificationResult, OptimizerSettings
from ddmo.utils.data_constructor import DataConstructor

BENCHMARK_FOLDER = Path(__file__).parent / "dataset_benchmark"

__all__ = [
    "BENCHMARK_FOLDER",
    "DataConstructor",
    "FormulationSettings",
    "IdentificationResult",
    "OptimizerSettings",
    "SolverSettings",
    "identify",
]
