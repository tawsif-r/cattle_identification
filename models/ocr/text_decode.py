
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

import os

import warnings

warnings.filterwarnings('ignore')

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