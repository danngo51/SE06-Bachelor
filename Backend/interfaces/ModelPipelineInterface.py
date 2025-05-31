from typing import Dict, Any, Optional, Union, List
import os
import pathlib
import pandas as pd
import numpy as np
import joblib
from abc import ABC, abstractmethod

class IModelPipeline(ABC):
    """
    Abstract base class for model pipelines.
    All model pipelines should inherit from this class and implement its methods.
    """
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for model input.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for model input
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Union[pd.Series, List[float]]:
        """
        Make predictions using the loaded model.
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load model from disk.
        
        Args:
            model_path: Path to saved model file
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass
