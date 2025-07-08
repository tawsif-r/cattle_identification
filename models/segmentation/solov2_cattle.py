import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import os
from scipy import ndimage
import matplotlib.pyplot as plt

class SOLOv2CattleSegmentation:
    """Enhanced SOLOv2 implementation for cattle segmentation"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['background', 'cattle_face', 'cattle_muzzle', 'cattle_ear_tag', 'cattle_body']
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),  # Standard SOLOv2 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load pre-trained model if path provided
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            print("Warning: No model path provided. Using enhanced image processing for demonstration")
            self.model = None
    
    def _load_model(self, model_path: str):
        """Load pre-trained SOLOv2 model"""
        try:
            # Load actual SOLOv2 model
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for SOLOv2 inference"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Apply transforms
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        return tensor
    
    def _postprocess_masks(self, masks: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output masks"""
        # Convert to numpy and resize to original dimensions
        masks_np = masks.cpu().numpy()
        processed_masks = []
        
        for mask in masks_np:
            # Resize mask to original image size
            resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
            processed_masks.append(resized_mask)
        
        return np.array(processed_masks)
    
    def segment_cattle(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment cattle image into different body parts"""
        height, width = image.shape[:2]
        
        if self.model is not None:
            # Use actual SOLOv2 model inference
            masks = self._inference_with_model(image)
        else:
            # Use enhanced image processing approach
            masks = self._create_intelligent_segmentation_masks(image)
        
        # Extract regions based on masks
        segmented_regions = {}
        for region_name, mask in masks.items():
            if region_name != 'background' and np.sum(mask) > 0:
                # Apply morphological operations to clean up mask
                kernel = np.ones((3, 3), np.uint8)
                mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
                
                # Apply mask and extract region
                masked_image = cv2.bitwise_and(image, image, mask=mask_cleaned)
                
                # Get bounding box with proper padding
                contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Find the largest contour (main region)
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Add adaptive padding based on region size
                    padding = max(10, min(w, h) // 10)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(width, x + w + padding)
                    y2 = min(height, y + h + padding)
                    
                    region = masked_image[y1:y2, x1:x2]
                    segmented_regions[region_name] = region
        
        return segmented_regions
    
    def _inference_with_model(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform inference using actual SOLOv2 model"""
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Post-process outputs
        masks = self._postprocess_masks(outputs['masks'], image.shape[:2])
        
        # Map masks to class names
        mask_dict = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(masks):
                mask_dict[class_name] = (masks[i] > 0.5).astype(np.uint8) * 255
        
        return mask_dict
    
    def _create_intelligent_segmentation_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Create intelligent segmentation masks using computer vision techniques"""
        height, width = image.shape[:2]
        masks = {}
        
        # Convert to different color spaces for better segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Use adaptive thresholding for better segmentation
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        # Face region detection (using color and texture analysis)
        face_mask = self._detect_face_region(image, hsv, lab)
        masks['cattle_face'] = face_mask
        
        # Muzzle region detection (darker, central area)
        muzzle_mask = self._detect_muzzle_region(image, hsv, gray, face_mask)
        masks['cattle_muzzle'] = muzzle_mask
        
        # Ear tag detection (bright, contrasting rectangular regions)
        ear_tag_mask = self._detect_ear_tag_region(image, hsv, edges)
        masks['cattle_ear_tag'] = ear_tag_mask
        
        return masks
    
    def _detect_face_region(self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Detect cattle face region using color and texture analysis"""
        height, width = image.shape[:2]
        
        # Color-based segmentation for cattle skin/fur
        # Cattle faces typically have specific color ranges
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([20, 255, 200])
        
        # Create mask for brown/tan colors
        color_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Use L*a*b* color space for better skin tone detection
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        
        # Adaptive thresholding on L channel
        face_candidate = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 15, 10)
        
        # Combine color and brightness information
        combined_mask = cv2.bitwise_and(color_mask, face_candidate)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        face_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        face_mask = self._remove_small_objects(face_mask, min_size=1000)
        
        return face_mask
    
    def _detect_muzzle_region(self, image: np.ndarray, hsv: np.ndarray, gray: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """Detect cattle muzzle region (typically darker, central area)"""
        # Muzzle is usually darker than surrounding face
        # Apply additional filtering within face region
        
        # Create a mask for darker regions
        _, dark_regions = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_regions = 255 - dark_regions  # Invert to get dark regions
        
        # Restrict to face region
        muzzle_candidate = cv2.bitwise_and(dark_regions, face_mask)
        
        # Look for circular/elliptical shapes (typical muzzle shape)
        contours, _ = cv2.findContours(muzzle_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        muzzle_mask = np.zeros_like(gray)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area for muzzle
                # Check if contour is roughly circular/elliptical
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.3:  # Reasonably circular
                    cv2.fillPoly(muzzle_mask, [contour], 255)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        muzzle_mask = cv2.morphologyEx(muzzle_mask, cv2.MORPH_CLOSE, kernel)
        
        return muzzle_mask
    
    def _detect_ear_tag_region(self, image: np.ndarray, hsv: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Detect ear tag region (bright, contrasting rectangular shapes)"""
        # Ear tags are typically bright, contrasting colors
        # Look for high saturation, high value regions
        
        # Create mask for bright, saturated regions
        lower_bright = np.array([0, 100, 150])
        upper_bright = np.array([180, 255, 255])
        bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # Combine with edge information
        combined = cv2.bitwise_and(bright_mask, edges)
        
        # Find rectangular contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ear_tag_mask = np.zeros_like(edges)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:  # Typical ear tag size range
                # Check if contour is roughly rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:  # Roughly rectangular
                    cv2.fillPoly(ear_tag_mask, [contour], 255)
        
        return ear_tag_mask
    
    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small objects from binary mask"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create output mask
        output_mask = np.zeros_like(mask)
        
        # Keep only components larger than min_size
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output_mask[labels == i] = 255
        
        return output_mask
    
    def visualize_segmentation(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Visualize segmentation results"""
        height, width = image.shape[:2]
        overlay = image.copy()
        
        # Define colors for each class
        colors = {
            'cattle_face': (0, 255, 0),      # Green
            'cattle_muzzle': (255, 0, 0),    # Blue
            'cattle_ear_tag': (0, 0, 255),   # Red
            'cattle_body': (255, 255, 0)     # Cyan
        }
        
        # Apply colored overlays
        for region_name, mask in masks.items():
            if region_name in colors and np.sum(mask) > 0:
                color = colors[region_name]
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = color
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        return overlay
    
    def get_segmentation_metrics(self, masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate segmentation quality metrics"""
        metrics = {}
        
        for region_name, mask in masks.items():
            if region_name != 'background':
                # Calculate basic metrics
                total_pixels = mask.shape[0] * mask.shape[1]
                segmented_pixels = np.sum(mask > 0)
                
                # Calculate connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                
                metrics[region_name] = {
                    'coverage_ratio': segmented_pixels / total_pixels,
                    'num_components': num_labels - 1,  # Exclude background
                    'largest_component_area': np.max(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0,
                    'average_component_area': np.mean(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0
                }
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Initialize segmentation system
    segmenter = SOLOv2CattleSegmentation()
    
    # Load and process image
    image_path = "cattle_image.jpg"  # Replace with your image path
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        
        # Perform segmentation
        segmented_regions = segmenter.segment_cattle(image)
        
        # Display results
        for region_name, region in segmented_regions.items():
            cv2.imshow(f"{region_name}", region)
        
        # Show metrics
        masks = segmenter._create_intelligent_segmentation_masks(image)
        metrics = segmenter.get_segmentation_metrics(masks)
        
        print("\nSegmentation Metrics:")
        for region, metric in metrics.items():
            print(f"{region}: {metric}")
        
        # Visualize overlay
        overlay = segmenter.visualize_segmentation(image, masks)
        cv2.imshow("Segmentation Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image not found: {image_path}")
