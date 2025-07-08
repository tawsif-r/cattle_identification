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