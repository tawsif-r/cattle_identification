# ===== Project Directory Structure =====
"""
cattle_identification_system/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── model_config.py
├── models/
│   ├── __init__.py
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── solov2_cattle.py
│   ├── face_recognition/
│   │   ├── __init__.py
│   │   ├── facenet_inception.py
│   │   └── facenet_mobile.py
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── ppocrv4.py
│   │   └── text_detection.py
│   └── base_model.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   └── transforms.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   └── metrics.py
├── inference/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── feature_extractor.py
│   └── matcher.py
├── database/
│   ├── __init__.py
│   ├── cattle_db.py
│   └── feature_storage.py
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   ├── visualization.py
│   └── logging_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_data.py
├── scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── run_inference.py
│   └── build_database.py
├── pretrained_models/
│   ├── facenet_inception_resnet_v1.pth
│   ├── facenet_mobilenet_v1.pth
│   ├── ppocrv4_det.pth
│   ├── ppocrv4_rec.pth
│   └── solov2_cattle.pth
├── sample_data/
│   ├── cattle_images/
│   ├── annotations/
│   └── ground_truth/
└── outputs/
    ├── features/
    ├── visualizations/
    └── logs/
"""
## Prompt help
```python 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os
import json
import sqlite3
from pathlib import Path
import warnings
from config.model_config import Config
warnings.filterwarnings('ignore')
from inference.feature_extractor import BaseFeatureExtractor
from models.segmentation.solov2_cattle import SOLOv2CattleSegmentation
from models.face_recognition.facenet_inception import FaceNetInceptionResNetV1
from models.face_recognition.facenet_mobile import FaceNetMobileNetV1
from models.ocr.text_decode import PPOCRV4TextRecognition,PPOCRV4EarTagRecognizer,PPOCRV4TextDetection
from database.database_manage import CattleDatabase
from inference.pipeline import CattleIdentificationPipeline

# # ===== Configuration =====
class Config:
    """Configuration class for the cattle identification system"""
    
    # Model configurations
    FACE_EMBEDDING_DIM = 128
    MUZZLE_EMBEDDING_DIM = 128
    FACE_MODEL_TYPE = "inception_resnet_v1"  # or "mobilenet_v1"
    
    # Training parameters
    TRIPLET_MARGIN = 0.5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]
    
    # Database
    DATABASE_PATH = "cattle_database.db"
    FEATURE_STORAGE_PATH = "features/"
    
    # Paths
    PRETRAINED_MODELS_PATH = "pretrained_models/"
    OUTPUT_PATH = "outputs/"

# ===== Base Model Class =====
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

# ===== SOLOv2 Cattle Segmentation =====
class SOLOv2CattleSegmentation:
    """Enhanced SOLOv2 implementation for cattle segmentation"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['background', 'cattle_face', 'cattle_muzzle', 'cattle_ear_tag', 'cattle_body']
        
        # Load pre-trained model if path provided
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            print("Using mock segmentation for demonstration")
            self.model = None
            
    def _load_model(self, model_path: str):
        """Load pre-trained SOLOv2 model"""
        # In practice, load actual SOLOv2 model
        # return torch.load(model_path, map_location=self.device)
        pass
        
    def segment_cattle(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment cattle image into different body parts"""
        height, width = image.shape[:2]
        
        # Create mock segmentation masks (replace with actual model inference)
        masks = self._create_enhanced_segmentation_masks(image)
        
        # Extract regions based on masks
        segmented_regions = {}
        for region_name, mask in masks.items():
            if region_name != 'background':
                # Apply mask and extract region
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                
                # Get bounding box
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(width, x + w + padding)
                    y2 = min(height, y + h + padding)
                    
                    region = masked_image[y1:y2, x1:x2]
                    segmented_regions[region_name] = region
        
        return segmented_regions
    
    def _create_enhanced_segmentation_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Create enhanced mock segmentation masks"""
        height, width = image.shape[:2]
        masks = {}
        
        # Enhanced face region (upper portion)
        face_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(face_mask, (width//2, height//3), (width//4, height//6), 0, 0, 360, 255, -1)
        masks['cattle_face'] = face_mask
        
        # Muzzle region (center, more precise)
        muzzle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(muzzle_mask, (width//2, height//2), (width//8, height//8), 0, 0, 360, 255, -1)
        masks['cattle_muzzle'] = muzzle_mask
        
        # Ear tag region (typically on ears)
        ear_tag_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(ear_tag_mask, (width//8, height//6), (width//4, height//4), 255, -1)
        masks['cattle_ear_tag'] = ear_tag_mask
        
        return masks

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

# ===== FaceNet MobileNet V1 for Muzzle Recognition =====
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

#===== PPOCRV4 for Ear Tag Recognition =====
class PPOCRV4TextDetection(nn.Module):
    """PPOCRV4 Text Detection Network (PP-LC NetV3 based)"""
    
    def __init__(self):
        super(PPOCRV4TextDetection, self).__init__()
        
        # Lightweight backbone (PP-LC NetV3 style)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Text detection head
        self.text_detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),  # Binary classification
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Detect text regions in ear tag"""
        features = self.backbone(x)
        text_map = self.text_detector(features)
        return text_map

class PPOCRV4TextRecognition(nn.Module):
    """PPOCRV4 Text Recognition Network"""
    
    def __init__(self, num_classes: int = 37):  # 0-9, A-Z, space
        super(PPOCRV4TextRecognition, self).__init__()
        
        # CNN backbone for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # RNN for sequence modeling (fix input size)
        self.rnn = nn.LSTM(256 * 8, 128, bidirectional=True, batch_first=True)  # 256 channels * 8 height = 2048
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Recognize text in detected regions"""
        # CNN feature extraction
        features = self.cnn(x)
        
        # Reshape for RNN: [B, C, H, W] -> [B, W, C*H]
        batch_size, channels, height, width = features.shape
        features = features.permute(0, 3, 1, 2)  # [B, W, C, H]
        features = features.reshape(batch_size, width, channels * height)  # [B, W, C*H]
        
        # RNN sequence modeling
        rnn_out, _ = self.rnn(features)
        
        # Classification
        output = self.classifier(rnn_out)
        return output

class PPOCRV4EarTagRecognizer:
    """Complete PPOCRV4 pipeline for ear tag recognition"""
    
    def __init__(self, det_model_path: str = None, rec_model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.text_detector = PPOCRV4TextDetection()
        self.text_recognizer = PPOCRV4TextRecognition()
        
        # Load pre-trained weights if available
        if det_model_path and os.path.exists(det_model_path):
            self.text_detector.load_state_dict(torch.load(det_model_path, map_location=self.device))
        
        if rec_model_path and os.path.exists(rec_model_path):
            self.text_recognizer.load_state_dict(torch.load(rec_model_path, map_location=self.device))
            
        self.text_detector.to(self.device)
        self.text_recognizer.to(self.device)
        
        # Character mapping
        self.char_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        
    def detect_and_recognize(self, ear_tag_image: np.ndarray) -> str:
        """Detect and recognize text in ear tag image"""
        # Preprocess image
        image_pil = Image.fromarray(ear_tag_image)
        transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        # Text detection
        with torch.no_grad():
            text_map = self.text_detector(image_tensor)
            
        # Text recognition
        with torch.no_grad():
            text_output = self.text_recognizer(image_tensor)
            
        # Decode text
        recognized_text = self._decode_text(text_output)
        return recognized_text
    
    def _decode_text(self, output: torch.Tensor) -> str:
        """Decode model output to text"""
        # Simple greedy decoding
        _, predicted = torch.max(output, 2)
        predicted = predicted.squeeze(0).cpu().numpy()
        
        # Convert to characters
        text = ""
        for idx in predicted:
            if idx < len(self.char_map):
                text += self.char_map[idx]
        
        return text.strip()

# ===== Triplet Loss Implementation =====
class TripletLoss(nn.Module):
    """Triplet Loss for learning discriminative features"""
    
    def __init__(self, margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        """Compute triplet loss"""
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()

# ===== Complete Pipeline =====
class CattleIdentificationPipeline:
    """Complete cattle identification pipeline"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.segmentation_model = SOLOv2CattleSegmentation()
        self.face_model = FaceNetInceptionResNetV1(config.FACE_EMBEDDING_DIM)
        self.muzzle_model = FaceNetMobileNetV1(config.MUZZLE_EMBEDDING_DIM)
        self.ear_tag_model = PPOCRV4EarTagRecognizer()
        
        # Move models to device
        self.face_model.to(self.device)
        self.muzzle_model.to(self.device)
        
        # Initialize losses
        self.triplet_loss = TripletLoss(config.TRIPLET_MARGIN)
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZATION_MEAN, std=config.NORMALIZATION_STD)
        ])
        
    def extract_all_features(self, image_path: str) -> Dict[str, any]:
        """Extract features from all cattle body parts"""
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Segment cattle
        print("Segmenting cattle image...")
        segmented_regions = self.segmentation_model.segment_cattle(image_rgb)
        
        results = {}
        
        # Step 2: Extract face features
        if 'cattle_face' in segmented_regions:
            print("Extracting face features...")
            face_region = segmented_regions['cattle_face']
            face_features = self._extract_face_features(face_region)
            results['face_features'] = face_features
            
        # Step 3: Extract muzzle features
        if 'cattle_muzzle' in segmented_regions:
            print("Extracting muzzle features...")
            muzzle_region = segmented_regions['cattle_muzzle']
            muzzle_features = self._extract_muzzle_features(muzzle_region)
            results['muzzle_features'] = muzzle_features
            
        # Step 4: Extract ear tag text
        if 'cattle_ear_tag' in segmented_regions:
            print("Recognizing ear tag text...")
            ear_tag_region = segmented_regions['cattle_ear_tag']
            ear_tag_text = self._extract_ear_tag_text(ear_tag_region)
            results['ear_tag_text'] = ear_tag_text
            
        results['segmented_regions'] = segmented_regions
        return results
    
    def _extract_face_features(self, face_region: np.ndarray) -> np.ndarray:
        """Extract face features using FaceNet Inception ResNet V1"""
        face_pil = Image.fromarray(face_region)
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        self.face_model.eval()
        with torch.no_grad():
            features = self.face_model(face_tensor)
        
        return features.cpu().numpy().squeeze()
    
    def _extract_muzzle_features(self, muzzle_region: np.ndarray) -> np.ndarray:
        """Extract muzzle features using FaceNet MobileNet V1"""
        muzzle_pil = Image.fromarray(muzzle_region)
        muzzle_tensor = self.transform(muzzle_pil).unsqueeze(0).to(self.device)
        
        self.muzzle_model.eval()
        with torch.no_grad():
            features = self.muzzle_model(muzzle_tensor)
        
        return features.cpu().numpy().squeeze()
    
    def _extract_ear_tag_text(self, ear_tag_region: np.ndarray) -> str:
        """Extract ear tag text using PPOCRV4"""
        return self.ear_tag_model.detect_and_recognize(ear_tag_region)
    
    def visualize_results(self, image_path: str, results: Dict[str, any], save_path: str = None):
        """Visualize extraction results"""
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmented regions
        segmented_regions = results['segmented_regions']
        region_names = ['cattle_face', 'cattle_muzzle', 'cattle_ear_tag']
        
        for idx, region_name in enumerate(region_names):
            if region_name in segmented_regions:
                axes[idx + 1].imshow(segmented_regions[region_name])
                axes[idx + 1].set_title(f'{region_name.replace("_", " ").title()}')
                axes[idx + 1].axis('off')
        
        # Feature information
        info_text = ""
        if 'face_features' in results:
            info_text += f"Face Features: {len(results['face_features'])}-dim vector\n"
        if 'muzzle_features' in results:
            info_text += f"Muzzle Features: {len(results['muzzle_features'])}-dim vector\n"
        if 'ear_tag_text' in results:
            info_text += f"Ear Tag Text: {results['ear_tag_text']}\n"
            
        axes[4].text(0.1, 0.5, info_text, fontsize=12, transform=axes[4].transAxes)
        axes[4].set_title('Extracted Information')
        axes[4].axis('off')
        
        # Hide unused subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ===== Database Management =====
class CattleDatabase:
    """Database for storing cattle features and information"""
    
    def __init__(self, db_path: str = "cattle_database.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cattle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cattle_id TEXT UNIQUE,
                name TEXT,
                breed TEXT,
                age INTEGER,
                gender TEXT,
                owner TEXT,
                registration_date TEXT,
                face_features BLOB,
                muzzle_features BLOB,
                ear_tag_text TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def store_cattle(self, cattle_info: Dict[str, any]):
        """Store cattle information and features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy arrays to bytes
        face_features_bytes = cattle_info['face_features'].tobytes() if 'face_features' in cattle_info else None
        muzzle_features_bytes = cattle_info['muzzle_features'].tobytes() if 'muzzle_features' in cattle_info else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO cattle 
            (cattle_id, name, breed, age, gender, owner, registration_date, 
             face_features, muzzle_features, ear_tag_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cattle_info['cattle_id'],
            cattle_info.get('name', ''),
            cattle_info.get('breed', ''),
            cattle_info.get('age', 0),
            cattle_info.get('gender', ''),
            cattle_info.get('owner', ''),
            cattle_info.get('registration_date', ''),
            face_features_bytes,
            muzzle_features_bytes,
            cattle_info.get('ear_tag_text', '')
        ))
        
        conn.commit()
        conn.close()
        
    def search_cattle(self, query_features: Dict[str, any], threshold: float = 0.7) -> List[Dict[str, any]]:
        """Search for cattle based on features"""
        # This would implement the matching logic using distance metrics
        # For now, return empty list
        return []

# ===== Usage Example =====
def main():
    """Example usage of the complete cattle identification system"""
    
    # Initialize pipeline
    pipeline = CattleIdentificationPipeline()
    
    # Initialize database
    db = CattleDatabase()
    
    # Example image path
    image_path = "cattle_image.jpg"  # Replace with actual image path
    
    try:
        # Extract all features
        print("Starting cattle identification pipeline...")
        results = pipeline.extract_all_features(image_path)
        
        # Display results
        print("\n=== Extraction Results ===")
        if 'face_features' in results:
            print(f"Face features: {len(results['face_features'])}-dimensional vector")
            print(f"Face features preview: {results['face_features'][:5]}...")
            
        if 'muzzle_features' in results:
            print(f"Muzzle features: {len(results['muzzle_features'])}-dimensional vector")
            print(f"Muzzle features preview: {results['muzzle_features'][:5]}...")
            
        if 'ear_tag_text' in results:
            print(f"Ear tag text: {results['ear_tag_text']}")
            
        # Visualize results
        pipeline.visualize_results(image_path, results)
        
        # Store in database (example)
        cattle_info = {
            'cattle_id': 'COW001',
            'name': 'Bessie',
            'breed': 'Holstein',
            'age': 3,
            'gender': 'Female',
            'owner': 'Farm Owner',
            'registration_date': '2025-01-01',
            **results
        }
        
        db.store_cattle(cattle_info)
        print("\nCattle information stored in database!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please provide a valid cattle image path")

if __name__ == "__main__":
    main()

print("\n=== Cattle Identification System Ready ===")
print("Components:")
print("1. SOLOv2 for cattle segmentation")
print("2. FaceNet Inception ResNet V1 for face recognition")
print("3. FaceNet MobileNet V1 for muzzle pattern recognition")
print("4. PPOCRV4 for ear tag text recognition")
print("5. Complete pipeline with database storage")
print("6. Triplet loss training capabilities")
```