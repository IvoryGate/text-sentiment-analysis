"""
统一管理项目路径
"""
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
CLEANED_DIR = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"

# Experiment directories
CONFIGS_DIR = ROOT / "configs"
EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = ROOT / "checkpoints"

# Research outputs
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"