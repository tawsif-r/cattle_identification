
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from inference.feature_extractor import BaseFeatureExtractor


class InceptionModule(nn.Module):
    """Inception module for concatenating multiple branches"""
    def __init__(self, branch1, branch2, branch3, branch4):
        super(InceptionModule, self).__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.branch3 = branch3
        self.branch4 = branch4
        
    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        # Concatenate along channel dimension
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

# ===== FaceNet Inception ResNet V1 for Face Recognition =====
class FaceNetInceptionResNetV1(BaseFeatureExtractor):
    """FaceNet with Inception ResNet V1 backbone for cattle face recognition"""
    
    def __init__(self, embedding_dim: int = 128):
        super(FaceNetInceptionResNetV1, self).__init__(embedding_dim)
        
        # Simplified Inception ResNet V1 architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Inception-like blocks with correct channel calculations
        self.inception_blocks = nn.Sequential(
            self._make_inception_block(128, 32, 64, 32, 32),  # Output: 32+64+32+32 = 160 channels
            self._make_inception_block(160, 64, 96, 32, 32),  # Output: 64+96+32+32 = 224 channels
            self._make_inception_block(224, 96, 128, 32, 32), # Output: 96+128+32+32 = 288 channels
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final embedding layer (updated input size)
        self.embedding = nn.Sequential(
            nn.Linear(288, 256),  # Changed from 512 to 288 (output of last inception block)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim)
        )
        
    def _make_inception_block(self, in_channels, out_1x1, out_3x3, out_5x5, pool_proj):
        """Create an Inception block"""
        # Branch 1: 1x1 conv
        branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU()
        )
        
        # Branch 2: 1x1 -> 3x3 conv
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_1x1, out_3x3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 3: 1x1 -> 5x5 conv (using two 3x3 convs)
        branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Branch 4: max pool -> 1x1 conv
        branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU()
        )
        
        return InceptionModule(branch1, branch2, branch3, branch4)
    
    def forward(self, x):
        """Forward pass through FaceNet Inception ResNet V1"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.inception_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        embeddings = self.embedding(x)
        return self.normalize_features(embeddings)