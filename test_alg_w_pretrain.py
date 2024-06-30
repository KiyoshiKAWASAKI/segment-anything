from segment_anything import SamPredictor, sam_model_registry
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch
import warnings
warnings.filterwarnings('ignore')

"""
# This is for SAM from GitHub Repo
# model_type = "vit_h"
# pretrain_model_path = "/project01/cvrl/jhuang24/sam_data/models/sam_vit_h_4b8939.pth"

test_img_path = "/project01/cvrl/jhuang24/australia-backup/data/imgs/2019-11-20-F1-0167_896_896.jpg"
mask_path = "/project01/cvrl/jhuang24/australia-backup/data/masks/2019-11-20-F1-0167_896_896.jpg"
prompt = "grass"

img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


sam = sam_model_registry[model_type](checkpoint=pretrain_model_path)

# predictor = SamPredictor(sam)
# predictor.set_image(img)
# masks, _, _ = predictor.predict(prompt)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)

print(masks)
"""

# This is for HuggingFace SAM
# pretrain_model_path = "/project01/cvrl/jhuang24/sam_data/finetune/finetune_sam_epoch_0001.pth"
pretrain_model_path = "/project01/cvrl/jhuang24/sam_data/finetune"

test_img_path = "/project01/cvrl/jhuang24/australia-backup/data/imgs/2019-11-20-F1-0167_896_896.jpg"
mask_path = "/project01/cvrl/jhuang24/australia-backup/data/masks/2019-11-20-F1-0167_896_896.jpg"
prompt = "grass"

# Setup model and processor
print("Setting up model and processor.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained(pretrain_model_path).to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")


# Process image and extract embeddings
print("Process image and extract embeddings.")
raw_image = Image.open(test_img_path).convert("RGB")
inputs = processor(raw_image, return_tensors="pt").to(device)
image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

# 


