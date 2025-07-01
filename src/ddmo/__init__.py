# src/ddmo/__init__.py

from pathlib import Path

from .main import identify

__all__ = ["identify"]

BENCHMARK_FOLDER = Path(__file__).parent / 'dataset_benchmark'
