import requests
import os

# Your Danbooru credentials
USERNAME = ""
API_KEY = ""

# Search settings
TAG = "scenery"  # Change this to your desired tag (e.g., anime landscapes)
LIMIT = 15000  # Number of images to download

# API request URL
url = f"https://danbooru.donmai.us/posts.json?tags={TAG}&limit={LIMIT}"

# Make request with authentication
response = requests.get(url, auth=(USERNAME, API_KEY))

if response.status_code == 200:
    posts = response.json()

    # Create folder to save images
    os.makedirs("danbooru_images", exist_ok=True)

    # Download images
    for i, post in enumerate(posts):
        if "file_url" in post:
            img_url = post["file_url"]
            img_data = requests.get(img_url).content
            
            # Save image
            filename = f"danbooru_images/image_{i}.jpg"
            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"Downloaded: {filename}")

    print("✅ Download Complete!")
else:
    print(f"❌ Error: {response.status_code} - {response.text}")