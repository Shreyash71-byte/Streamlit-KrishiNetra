import os
import time
from icrawler.builtin import BingImageCrawler

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------

IMAGE_LIMIT = 20
BASE_DATA_DIR = "02_dataset/00_downloaded_raw"


# ----------------------------------------------------
# YOUR CURRENT CLASSES
# ----------------------------------------------------
PLANT_CLASSES = {

# ================= APPLE =================

"Apple___Apple_scab":
"closeup photo of apple leaf only infected with apple scab disease symptoms",

"Apple___Black_rot":
"closeup photo of apple leaf only infected with black rot disease symptoms",

"Apple___Cedar_apple_rust":
"closeup photo of apple leaf only infected with cedar apple rust disease symptoms",

"Apple___healthy":
"closeup photo of healthy apple leaf only",

# ================= CORN =================

"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
"closeup photo of corn leaf only infected with gray leaf spot disease symptoms",

"Corn_(maize)__*Common_rust*":
"closeup photo of corn leaf only infected with common rust disease symptoms",

"Corn_(maize)___healthy":
"closeup photo of healthy corn leaf only",

"Corn_(maize)___Northern_Leaf_Blight":
"closeup photo of corn leaf only infected with northern leaf blight disease symptoms",

# ================= GRAPE =================

"Grape___Black_rot":
"closeup photo of grape leaf only infected with black rot disease symptoms",

"Grape___Esca_(Black_Measles)":
"closeup photo of grape leaf only infected with esca disease symptoms",

"Grape___healthy":
"closeup photo of healthy grape leaf only",

"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
"closeup photo of grape leaf only infected with leaf blight disease symptoms",

# ================= POTATO =================

"Potato___Early_blight":
"closeup photo of potato leaf only infected with early blight disease symptoms",

"Potato___healthy":
"closeup photo of healthy potato leaf only",

"Potato___Late_blight":
"closeup photo of potato leaf only infected with late blight disease symptoms",

# ================= TOMATO =================

"Tomato___Early_blight":
"closeup photo of tomato leaf only infected with early blight disease symptoms",

"Tomato___healthy":
"closeup photo of healthy tomato leaf only",

"Tomato___Leaf_Mold":
"closeup photo of tomato leaf only infected with leaf mold disease symptoms",

"Tomato___Septoria_leaf_spot":
"closeup photo of tomato leaf only infected with septoria leaf spot disease symptoms"

}



# ----------------------------------------------------
# DOWNLOAD FUNCTION
# ----------------------------------------------------

def download_class_images(folder_name, search_query):

    folder_path = os.path.join(
        BASE_DATA_DIR,
        folder_name
    )

    os.makedirs(folder_path, exist_ok=True )

    existing_images = len([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
    ])

    if existing_images >= IMAGE_LIMIT:
        print(f"Skipping {folder_name}" )
        return
    
    remaining = IMAGE_LIMIT - existing_images


    print(f"\nDownloading {remaining} images for:\n{folder_name}" )

    crawler = BingImageCrawler(storage = {'root_dir': folder_path}, downloader_threads=4)

    crawler.crawl(keyword = search_query, max_num = remaining)
    print(f"Done -> {folder_name}" )
    time.sleep(2)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__ == "__main__":
    print("\nStarting dataset expansion...\n")

    for folder, query in PLANT_CLASSES.items():

        download_class_images(
            folder,
            query
        )

    print("\nAll classes processed.\n")
