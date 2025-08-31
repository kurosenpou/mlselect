#!/usr/bin/env python3
"""
Algorithm Manager - Unified interface for all ML algorithms

This module provides a unified interface to access regression, classification,
and clustering algorithms in a modular way.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union

from .regression import RegressionAlgorithms
from .classification import ClassificationAlgorithms
from .clustering import ClusteringAlgorithms


class AlgorithmManager:
    """
    Unified manager for all ML algorithms.
    Provides a single interface to access regression, classification, and clustering algorithms.
    """
    
    def __init__(self):
        self.regression = RegressionAlgorithms()
        self.classification = ClassificationAlgorithms()
        self.clustering = ClusteringAlgorithms()
        
        # Algorithm type mapping
        self.algorithm_types = {
            # Regression algorithms
            'linear_regression': 'regression',
            'polynomial_regression': 'regression',
            'random_forest_regression': 'regression',
            'svr': 'regression',
            'decision_tree_regression': 'regression',
            
            # Classification algorithms
            'logistic_regression': 'classification',
            'random_forest_classification': 'classification',
            'svm': 'classification',
            'knn': 'classification',
            'naive_bayes': 'classification',
            'decision_tree_classification': 'classification',
            
            # Clustering algorithms
            'kmeans': 'clustering',
            'hierarchical_clustering': 'clustering'
        }
    
    def get_algorithm_type(self, algorithm_name: str) -> str:
        """
        Get the type of algorithm (regression, classification, clustering)
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm type string
        """
        return self.algorithm_types.get(algorithm_name.lower(), 'unknown')
    
    def get_algorithm_engine(self, algorithm_name: str):
        """
        Get the appropriate algorithm engine based on algorithm name
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm engine instance
        """
        alg_type = self.get_algorithm_type(algorithm_name)
        
        if alg_type == 'regression':
            return self.regression
        elif alg_type == 'classification':
            return self.classification
        elif alg_type == 'clustering':
            return self.clustering
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    feature_columns: list, algorithm: str = None) -> Dict[str, Any]:
        """
        Prepare data for training using the appropriate algorithm engine
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            feature_columns: List of feature column names
            algorithm: Algorithm name (optional)
            
        Returns:
            Prepared data dictionary
        """
        if algorithm:
            engine = self.get_algorithm_engine(algorithm)
            return engine.prepare_data(df, target_column, feature_columns, algorithm)
        else:
            # Default to regression engine for backward compatibility
            return self.regression.prepare_data(df, target_column, feature_columns)
    
    def train_model(self, algorithm_name: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a model using the appropriate algorithm engine
        
        Args:
            algorithm_name: Name of the algorithm to use
            data_dict: Prepared data dictionary
            
        Returns:
            Training result dictionary
        """
        engine = self.get_algorithm_engine(algorithm_name)
        return engine.train_model(algorithm_name, data_dict)
    
    def predict(self, model_id: str, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Args:
            model_id: ID of the trained model
            data: Input data for prediction
            
        Returns:
            Prediction results
        """
        # Try to find the model in each engine
        for engine in [self.regression, self.classification, self.clustering]:
            try:
                result = engine.predict(model_id, data)
                if result and 'error' not in result:
                    return result
            except:
                continue
        
        return {'error': f'Model {model_id} not found in any engine'}
    
    def get_available_algorithms(self) -> Dict[str, list]:
        """
        Get all available algorithms grouped by type

        Returns:
            Dictionary with algorithm types as keys and algorithm lists as values
        """
        return {
            'regression': self.regression.get_supported_algorithms(),
            'classification': self.classification.get_supported_algorithms(),
            'clustering': self.clustering.get_supported_algorithms()
        }
    
    def get_all_algorithms(self) -> list:
        """
        Get a flat list of all available algorithms
        
        Returns:
            List of all algorithm names
        """
        all_algorithms = []
        available = self.get_available_algorithms()
        for alg_type, algorithms in available.items():
            all_algorithms.extend(algorithms)
        return all_algorithms