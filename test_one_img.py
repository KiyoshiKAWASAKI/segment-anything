import json
import pickle
import os, sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import SamModel
from transformers import SamProcessor

from datasets import load_dataset

from torch.optim import Adam
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize

from segdataset import SegmentationDataset
from statistics import mean
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import jaccard_score

import warnings
warnings.filterwarnings('ignore')



# Data and model directories
# data_dir = "/project01/cvrl/jhuang24/alg_seg_vis/vis_img"
data_dir = "/project01/cvrl/jhuang24/australia-backup/data"
save_model_path = "/project01/cvrl/jhuang24/sam_data/finetune"
save_vis_path = "/project01/cvrl/jhuang24/alg_seg_vis/vis_results/sam"


batch_size = 1
threshold = 0.5



def test_model(data_directory,
               model_dir,
               save_vis_path,
               batch_size,
               threshold):
    """

    :param data_directory:
    :param exp_directory:
    :param batch_size:
    :param threshold:
    :return:
    """
    # Load pretrain model
    model = SamModel.from_pretrained(model_dir)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda:0')
    model.to(device)

    # Define customized dataset
    data_transforms = transforms.Compose([transforms.ToTensor()])

    dataset = SegmentationDataset(root=data_directory,
                                    processor=processor,
                                    image_folder='imgs',
                                    mask_folder='masks',
                                    transforms=data_transforms)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)


    for batch in tqdm(dataloader):
        with torch.no_grad():
            # Get predictions

            outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_boxes=batch["input_boxes"].to(device),
                          multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            img_path = batch["img_path"]
            img_name = img_path[0].split("/")[-1]
            # print(img_name)

            # print(predicted_masks.shape) # torch.Size([1, 1, 256, 256])
            # print(ground_truth_masks.shape) # torch.Size([1, 1, 256, 256])
            # print(predicted_masks)
            # print(ground_truth_masks)
            # sys.exit()

            # Convert logits to probability
            output_prob = torch.nn.Sigmoid()(predicted_masks)

            # print(output_prob)

            # Convert everything into numpy
            prob_np = output_prob.cpu().numpy()
            mask_np = ground_truth_masks.cpu().numpy()
            mask_final = np.where(mask_np == 1.0, 1.0, 0.0)

            # Thresholding probabilities
            prob_final = np.where(prob_np > threshold, 1.0, 0.0)
            # print(np.unique(prob_final))
            #
            # print(np.squeeze(mask_final).shape)
            # print(np.squeeze(prob_final).shape)

            pred_img = Image.fromarray(np.squeeze(prob_final)*255)
            pred_img = pred_img.convert('RGB')
            pred_img.save(os.path.join(save_vis_path, img_name))




if __name__ == "__main__":
    test_model(data_directory=data_dir,
               model_dir=save_model_path,
               save_vis_path=save_vis_path,
               batch_size=batch_size,
               threshold=threshold)


