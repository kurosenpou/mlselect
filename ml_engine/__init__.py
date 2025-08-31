"""ML Engine - 机器学习引擎模块"""

__version__ = "1.0.0"
__author__ = "ML Select Team"

# 导入主要模块
from . import algorithms
from . import evaluation
from . import feature_analysis
from . import model_selection

__all__ = [
    "algorithms",
    "evaluation", 
    "feature_analysis",
    "model_selection"
]