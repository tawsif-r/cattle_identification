import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from database_client import DatabaseClient
from inference_sdk import InferenceHTTPClient
import json

def InferenceResult():
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="vHg5bmHnCP2sJJWxC1ht"
    )

    result = client.run_workflow(
        workspace_name="brac-6bgtt",
        workflow_id="custom-workflow",
        images={
            "image": "cow_7_test.jpg"
        },
        use_cache=True # cache workflow definition for 15 minutes
    )

    print("=== RAW WORKFLOW RESULT ===")
    # output = json.dumps(result, indent=2)
    segmentation_result = result[0].get("predictions","").get("predictions",[])[0].get("points",[])
    return segmentation_result

def create_mask_from_polygon(image_size, points):
    """
    Create a binary mask from polygon points.
    Args:
        image_size: Tuple of (width, height) of the image
        points: List of dictionaries with 'x' and 'y' coordinates
    Returns:
        PIL Image: Binary mask (white for muzzle, black for background)
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    polygon = [(p['x'], p['y']) for p in points]
    draw.polygon(polygon, fill=255)
    return mask

def crop_muzzle_region(image_path, mask):
    """
    Crop the muzzle region using the mask.
    Args:
        image_path: Path to the original image
        mask: PIL Image mask
    Returns:
        PIL Image: Cropped muzzle region
    """
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # Apply mask to get muzzle region
    masked_image = image_np.copy()
    masked_image[mask_np == 0] = 0  # Set background to black
    
    # Find bounding box of the mask
    coords = np.where(mask_np > 0)
    if len(coords[0]) == 0:
        raise ValueError("No muzzle region found in mask")
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Crop the image
    cropped_image = Image.fromarray(masked_image[y_min:y_max+1, x_min:x_max+1])
    return cropped_image

def extract_features(image, model, transform, device):
    """
    Extract features from an image using a pretrained model.
    Args:
        image: PIL Image
        model: Pretrained PyTorch model
        transform: Image transformation pipeline
        device: torch.device (cpu or cuda)
    Returns:
        numpy array: Extracted features
    """
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu().numpy()

def visualize_and_save(image, output_path):
    """
    Save an image to file without attempting to display it.
    Args:
        image: PIL Image
        output_path: Path to save the image
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Muzzle Segment")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the figure to prevent display attempts
    print(f"Cropped muzzle saved to: {output_path}")

#TODO: make a database to save the features extracted from the muzzle with a reference number.

def main():
    # Initialize database client
    db_client = DatabaseClient()

    # Image path
    image_path = "cow_7_test.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image to get size
    image = Image.open(image_path).convert('RGB')
    image_size = image.size  # (width, height)

    # Create mask from polygon points
    # muzzle_points = segmentation_result[0]["points"]
    muzzle_points = InferenceResult()
    mask = create_mask_from_polygon(image_size, muzzle_points)

    # Crop the muzzle region
    cropped_muzzle = crop_muzzle_region(image_path, mask)

    # Visualize and save the cropped muzzle
    visualize_and_save(cropped_muzzle, "cropped_muzzle.png")

    # Load pretrained ResNet-50 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    
    # Remove the final fully connected layer to get features
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract features
    features = extract_features(cropped_muzzle, model, transform, device)
    print(f"Extracted features shape: {features.shape}")
    print(f"First few feature values: {features[:10]}")

    reference_number = "COW_007"

    db_client.save_features(reference_number,features)

    # Check if features match for the given reference number
    is_match, distance = db_client.match_features(reference_number, features, threshold=8.0)
    if is_match:
        print(f"Features match for reference number {reference_number} with distance: {distance:.4f}")
    else:
        if distance is None:
            print(f"No match or error for reference number {reference_number}")
        else:
            print(f"Features do not match for reference number {reference_number}. Distance: {distance:.4f}")

    # Save features to a file
    np.save("muzzle_features.npy", features)
    print("Features saved to: muzzle_features.npy")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure 'cow_face.jpg' exists in the current directory")
        print("2. Verify the segmentation points form a valid polygon")
        print("3. Check that torch, torchvision, PIL, numpy, and matplotlib are installed")
        print("4. Ensure sufficient memory for GPU/CPU operations")