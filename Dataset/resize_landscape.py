from PIL import Image
import os
from multiprocessing import Pool

input_root = "landscape"
output_folder = "resized_images"
os.makedirs(output_folder, exist_ok=True)

def resize_image(img_path):
    try:
        # Extract filename without the full path
        filename = os.path.basename(img_path)
        
        # Open image safely
        with Image.open(img_path) as img:
            # Convert image to RGB if needed
            if img.mode not in ("RGB", "L"):  
                img = img.convert("RGB")  # Converts P, RGBA, etc., to RGB
            
            # Resize image
            img = img.resize((256, 256), Image.LANCZOS)

            # Save the image in JPEG format
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
            img.save(output_path, "JPEG", quality=100)
        
    except Exception as e:
        print(f"Error resizing {img_path}: {e}")

if __name__ == "__main__":
    # Walk through all subdirectories and collect image paths
    images = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(("jpg", "png", "jpeg", "gif")):
                images.append(os.path.join(root, f))
    
    # Process images in parallel
    with Pool(os.cpu_count()) as p:
        p.map(resize_image, images)