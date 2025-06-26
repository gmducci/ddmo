# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# long description from your README
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="ddmo",
    version="0.1.0",
    description=(
        "Data-Driven Model Optimization: "
        "Discover differential equations from data using sparse regression "
        "and multi-objective optimization."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ddmo",
    license="MIT",
    python_requires=">=3.7, <4",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
        "matplotlib>=3.3",
        "pysindy>=0.7",
        "pymoo>=0.6",
        "statsmodels>=0.12",
        "PyWavelets>=1.1",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            # installs a `ddmo` command that runs ddmo.main:main()
            "ddmo=ddmo.main:main",
        ],
    },
)
