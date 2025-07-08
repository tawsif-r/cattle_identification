import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


class BaseFeatureExtractor(nn.Module):
    """Base class for all feature extractors"""
    
    def __init__(self, embedding_dim: int = 128):
        super(BaseFeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor"""
        raise NotImplementedError
        
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize features"""
        return F.normalize(features, p=2, dim=1)