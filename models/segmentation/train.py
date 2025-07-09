import os
from PIL import Image
import json

image_dir = "/home/tawsif/Documents/cattle_pipeline/models/segmentation/dataset/images"
label_dir = "/home/tawsif/Documents/cattle_pipeline/models/segmentation/dataset/label"

def load_images_and_label(image_dir,label_dir):
    # lists to store images and labels
    images = []
    labels = []

    print(os.listdir(image_dir))
    for img_file in os.listdir(image_dir):
        # Extract base name from image file
        base_name =  os.path.splitext(img_file)[0]

        # get labels
        label_file = f"{base_name}.json"
        label_path = os.path.join(label_dir,label_file)
        if os.path.exists(label_path):
            img_path = os.path.join(image_dir,img_file)
            image = Image.open(img_path)
            images.append(image)

            # load label
            with open(label_path,'r') as file:
                label = json.load(file) # read the file in label path
                labels.append(label)
        else:
            print(f"Label file {label_file} not found for image {img_file}")
    
    return images, labels

# load the data
images, labels = load_images_and_label(image_dir,label_dir)
print(f"Loaded {len(images)} images and {len(labels)} labels")

# if __name__ == "__main__":
#     load_images_and_label(image_dir,label_dir)
