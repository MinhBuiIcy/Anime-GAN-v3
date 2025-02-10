from PIL import Image
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
import numpy as np


input_folder = ""
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

def blur_edges(image):
    # Convert to grayscale to detect edges more easily
    gray_image = image.convert("L")
    
    # Apply an edge detection filter (Sobel)
    edge_image = gray_image.filter(ImageFilter.FIND_EDGES)

    # Invert the edge image to make edges white on black background
    edge_image = ImageOps.invert(edge_image)

    # Apply a Gaussian Blur to the edge image to soften the edges
    blurred_edges = edge_image.filter(ImageFilter.GaussianBlur(radius=10))

    # Ensure that the edge image is in RGB mode
    blurred_edges = blurred_edges.convert("L")

    # Apply a Gaussian Blur to the original image to blur the edges
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=15))

    # Ensure the original image is in RGB mode, even if it has an alpha channel
    image = image.convert("RGB")

    # Composite the original image and the blurred image using the edge mask
    final_image = Image.composite(image, blurred_image, blurred_edges)

    return final_image

def resize_image(filename):
    try:
        img_path = os.path.join(input_folder, filename)
        
        # Open image safely
        with Image.open(img_path) as img:
            # Convert image to RGB if needed
            if img.mode not in ("RGB", "L"):  
                img = img.convert("RGB")  # Converts P, RGBA, etc., to RGB
            
            # Resize image
            img = blur_edges(img)

            # Save the image in JPEG format
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
            img.save(output_path, "JPEG", quality=100)
        
    except Exception as e:
        print(f"Error resizing {filename}: {e}")

if __name__ == "__main__":
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(("jpg", "png", "jpeg", "gif"))]
    with Pool(os.cpu_count()) as p:
        p.map(resize_image, images)