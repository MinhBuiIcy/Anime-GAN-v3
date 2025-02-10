from openimages.download import download_images

# Define your classes and destination folder
classes = ["Scissors"]
destination_folder = "t"

# Download images
download_images(destination_folder, classes, 'exclusions.txt', limit=1, annotation_format=None)