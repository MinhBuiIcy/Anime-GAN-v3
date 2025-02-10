import requests
import os
import time

# Your Danbooru credentials
USERNAME = ""
API_KEY = ""

# Search settings
TAG = "scenery"  # Change to any tag (e.g., anime backgrounds)
TOTAL_IMAGES = 15000  # Total number of images you want
IMAGES_PER_REQUEST = 100  # Maximum is 100 per request

# Create folder to save images
os.makedirs("danbooru_images_15000_429", exist_ok=True)

downloaded = 0
page = 1  # Start from page 1

while downloaded < TOTAL_IMAGES:
    # API request URL with pagination
    url = f"https://danbooru.donmai.us/posts.json?tags={TAG}&limit={IMAGES_PER_REQUEST}&page={page}"
    try:
    # Make request with authentication
        response = requests.get(url, auth=(USERNAME, API_KEY))

        if response.status_code == 200:
            posts = response.json()

            if not posts:
                print("No more images found.")
                break  # Stop if no more results

            for post in posts:
                if "file_url" in post and downloaded < TOTAL_IMAGES:
                    img_url = post["file_url"]
                    try:
                        img_data = requests.get(img_url, timeout=10).content
                        filename = f"danbooru_images/image_{downloaded}.jpg"
                        with open(filename, "wb") as f:
                            f.write(img_data)
                        print(f"Downloaded {downloaded + 1}: {filename}")
                        downloaded += 1
                    except requests.exceptions.RequestException:
                        print(f"⚠️ Failed to download {img_url}, skipping...")

            page += 1  # Go to the next page
            time.sleep(1)  # Sleep to avoid getting blocked

        elif response.status_code == 429:
            print("Too Many Requests! Waiting 60 seconds...")
            time.sleep(30)
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            break
    except requests.exceptions.RequestException:
        print("Network error! Waiting 30 seconds before retrying...")
        time.sleep(30)

print("✅ Download Complete!")