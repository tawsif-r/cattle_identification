import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import warnings
warnings.filterwarnings('ignore')
from inference.feature_extractor import BaseFeatureExtractor

class FaceNetMobileNetV1(BaseFeatureExtractor):
    """FaceNet with MobileNet V1 backbone for cattle muzzle pattern recognition"""
    
    def __init__(self, embedding_dim: int = 128):
        super(FaceNetMobileNetV1, self).__init__(embedding_dim)
        
        # Use MobileNetV2 as base (V1 is less common in torchvision)
        self.backbone = mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Custom embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        """Forward pass through FaceNet MobileNet V1"""
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return self.normalize_features(embeddings)
