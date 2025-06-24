"""
Setup configuration for time-series-ensemble package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Fallback to core requirements
    requirements = [
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "structlog>=23.0.0",
        "crewai>=0.1.0"
    ]

setup(
    name="time-series-ensemble",
    version="0.1.0",
    description="Multi-Agent Time Series Forecasting Ensemble with Local LLM Orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Time Series Ensemble Team",
    author_email="team@timeseriesensemble.com",
    url="https://github.com/yourusername/time-series-ensemble",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "ruff>=0.0.260",
        ],
        "gpu": [
            "torch>=2.0.0",
            "tensorflow>=2.13.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ts-ensemble=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["time-series", "forecasting", "multi-agent", "ensemble", "machine-learning"],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/time-series-ensemble/issues",
        "Source": "https://github.com/yourusername/time-series-ensemble",
        "Documentation": "https://time-series-ensemble.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)
