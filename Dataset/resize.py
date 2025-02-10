from PIL import Image
import os
from multiprocessing import Pool

input_folder = "danbooru_images"
output_folder = "resized_images"
os.makedirs(output_folder, exist_ok=True)

def resize_image(filename):
    try:
        img_path = os.path.join(input_folder, filename)
        
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
        print(f"Error resizing {filename}: {e}")

if __name__ == "__main__":
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(("jpg", "png", "jpeg", "gif"))]
    with Pool(os.cpu_count()) as p:
        p.map(resize_image, images)