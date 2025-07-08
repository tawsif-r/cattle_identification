
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict
import warnings
from config.model_config import Config
warnings.filterwarnings('ignore')

from models.segmentation.solov2_cattle import SOLOv2CattleSegmentation
from models.face_recognition.facenet_inception import FaceNetInceptionResNetV1
from models.face_recognition.facenet_mobile import FaceNetMobileNetV1
from models.ocr.text_decode import PPOCRV4EarTagRecognizer
from distance_matching.matching import VotingClassifier,CattleMatcher
from database.database_manage import CattleDatabase
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
    
    def identify_cattle(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, any]:
        """Complete cattle identification pipeline with database matching"""
        
        # Step 1: Extract features
        print("Extracting features from cattle image...")
        features = self.extract_all_features(image_path)
        
        # Step 2: Initialize matcher and voting classifier
        matcher = CattleMatcher(CattleDatabase())
        voting_classifier = VotingClassifier()
        
        # Step 3: Match features against database
        print("Matching features against database...")
        match_results = matcher.match_features(features, top_k=3)
        
        # Step 4: Make final prediction using voting
        print("Making final prediction...")
        prediction = voting_classifier.predict(match_results, confidence_threshold)
        
        # Step 5: Add detailed analysis
        prediction['extracted_features'] = {
            'face_features_dim': len(features.get('face_features', [])),
            'muzzle_features_dim': len(features.get('muzzle_features', [])),
            'ear_tag_text': features.get('ear_tag_text', 'Not detected')
        }
        
        prediction['match_details'] = match_results
        prediction['segmented_regions'] = features.get('segmented_regions',{})
        
        return prediction
        
    def print_identification_results(self, results: Dict[str, any]):
        """Pretty print identification results"""
        print("\n" + "="*60)
        print("CATTLE IDENTIFICATION RESULTS")
        print("="*60)
        
        print(f"Status: {results['status']}")
        print(f"Predicted ID: {results['predicted_id']}")
        print(f"Confidence: {results['confidence']:.3f}")
        
        print(f"\nExtracted Features:")
        features = results['extracted_features']
        print(f"  - Face features: {features['face_features_dim']} dimensions")
        print(f"  - Muzzle features: {features['muzzle_features_dim']} dimensions")
        print(f"  - Ear tag text: {features['ear_tag_text']}")
        
        print(f"\nTop Candidates:")
        for cattle_id, score in list(results['all_candidates'].items())[:5]:
            print(f"  {cattle_id}: {score:.3f}")
        
        print(f"\nDetailed Match Results:")
        for cattle_id, details in results['detailed_results'].items():
            print(f"  {cattle_id} (Final: {details['final_score']:.3f}):")
            for modality, score in details['modality_scores'].items():
                print(f"    - {modality}: {score:.3f}")
        
        print("="*60)
        """Visualize extraction results"""
        # Load original image
        image_path="cattle_image.jpg"
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
        save_path = "output"
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_results(self, image_path: str, results: Dict[str, any], save_path: str = None):
        """Visualize identification results"""
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Original image with prediction overlay
        axes[0].imshow(image_rgb)
        prediction_text = f"ID: {results.get('predicted_id', 'Unknown')}\nConfidence: {results.get('confidence', 0):.3f}"
        axes[0].text(10, 30, prediction_text, fontsize=14, color='red', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        axes[0].set_title('Identified Cattle')
        axes[0].axis('off')
        
        # Segmented regions if available
        if 'segmented_regions' in results:
            segmented_regions = results['segmented_regions']
            region_names = ['cattle_face', 'cattle_muzzle', 'cattle_ear_tag']
            
            for idx, region_name in enumerate(region_names):
                if region_name in segmented_regions and idx + 1 < len(axes):
                    axes[idx + 1].imshow(segmented_regions[region_name])
                    axes[idx + 1].set_title(f'{region_name.replace("_", " ").title()}')
                    axes[idx + 1].axis('off')
        
        # Match results visualization
        if 'match_details' in results:
            match_details = results['match_details']
            
            # Create bar chart of top matches
            axes[4].clear()
            if results.get('all_candidates'):
                top_5 = list(results['all_candidates'].items())[:5]
                cattle_ids = [item[0] for item in top_5]
                scores = [item[1] for item in top_5]
                
                bars = axes[4].bar(range(len(cattle_ids)), scores)
                axes[4].set_xlabel('Cattle ID')
                axes[4].set_ylabel('Confidence Score')
                axes[4].set_title('Top 5 Matches')
                axes[4].set_xticks(range(len(cattle_ids)))
                axes[4].set_xticklabels(cattle_ids, rotation=45)
                
                # Color the best match differently
                if bars:
                    bars[0].set_color('green')
                    for bar in bars[1:]:
                        bar.set_color('lightblue')
            
            axes[4].grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()