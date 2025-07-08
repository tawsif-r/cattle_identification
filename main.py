import warnings
warnings.filterwarnings('ignore')
from database.database_manage import CattleDatabase
from inference.pipeline import CattleIdentificationPipeline
import numpy as np

# ===== Usage Example =====
def main():
    """Example usage with database matching and prediction"""
    
    # Initialize pipeline
    pipeline = CattleIdentificationPipeline()
    
    # Initialize database
    db = CattleDatabase()
    
    # Example: Add some cattle to database first
    print("Adding sample cattle to database...")
    
    # You would replace these with actual cattle images and data
    sample_cattle = [
        {
            'cattle_id': 'COW001',
            'name': 'Bessie',
            'breed': 'Holstein',
            'face_features': np.random.randn(128).astype(np.float32),  # Mock features
            'muzzle_features': np.random.randn(128).astype(np.float32),
            'ear_tag_text': 'A123'
        },
        {
            'cattle_id': 'COW002', 
            'name': 'Moobert',
            'breed': 'Angus',
            'face_features': np.random.randn(128).astype(np.float32),
            'muzzle_features': np.random.randn(128).astype(np.float32),
            'ear_tag_text': 'B456'
        },
        {
            'cattle_id': 'COW003',
            'name': 'Daisy',
            'breed': 'Jersey', 
            'face_features': np.random.randn(128).astype(np.float32),
            'muzzle_features': np.random.randn(128).astype(np.float32),
            'ear_tag_text': 'C789'
        }
    ]
    
    for cattle in sample_cattle:
        db.store_cattle(cattle)
    
    print("Sample cattle stored in database!")
    
    # Example identification
    image_path = "cattle_image.jpg"  # Replace with actual image path

    # TODO: make a module that will save the features of the image image 
    
    try:
        # Run complete identification pipeline
        print(f"\nIdentifying cattle from: {image_path}")
        results = pipeline.identify_cattle(image_path,confidence_threshold=0.3)
        
        # Print detailed results
        pipeline.print_identification_results(results)
        
        # Visualize results
        pipeline.visualize_results(image_path, results)
        
    except Exception as e:
        print(f"Error during identification: {e}")
        print("Make sure you have a valid cattle image at the specified path")


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