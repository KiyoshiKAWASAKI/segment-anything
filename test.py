import json
import pickle
import os, sys
import numpy as np
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
data_dir = "/project01/cvrl/jhuang24/australia-backup/data"
save_model_path = "/project01/cvrl/jhuang24/sam_data/finetune"


batch_size = 1
threshold = 0.5




def calculate_eer(fpr, tpr, thresholds):
    """

    :param fpr:
    :param tpr:
    :param thresholds:
    :return:
    """
    fnr = 1 - tpr
    # Find the nearest point where FPR equals FNR
    eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_threshold_index]
    eer = fpr[eer_threshold_index]

    return eer, eer_threshold, eer_threshold_index




def test_model(data_directory,
               model_dir,
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

    # Check shape of the data
    # batch = next(iter(dataloader))
    #
    # for k,v in batch.items():
    #   print(k,v.shape)
    # print("GT mask: ", batch["ground_truth_mask"].shape)

    # Iterate over test data and calculate metrics
    correct = 0
    total = 0
    jaccard = 0
    nb_sample = 0

    gt = []
    pred = []

    for batch in tqdm(dataloader):
        with torch.no_grad():
            nb_sample += 1
            # print("*" *20)

            # Get predictions
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_boxes=batch["input_boxes"].to(device),
                          multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

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

            gt.append(mask_final.flatten())
            pred.append(prob_np.flatten())

            """
            Calculate pixel wise accuracy and Jaccard Index.
            """
            # Thresholding probabilities
            prob_final = np.where(prob_np > threshold, 1.0, 0.0)

            # print(np.unique(prob_final), np.unique(mask_final))

            # Correct is torch tensor, total is int - JH
            correct += (torch.flatten(torch.tensor(prob_final)) == torch.flatten(torch.tensor(mask_final))).sum()
            total += len(torch.flatten(torch.tensor(prob_final)))
            # print(correct, total)

            jaccard += jaccard_score(np.squeeze(mask_final),
                                     np.squeeze(prob_final),
                                     average="micro")

    # Obtain final accuracy
    accuracy = correct.detach().numpy() / float(total)

    # Obtain final Jaccard score
    jaccard = jaccard / nb_sample

    # Calculate Dice Score
    dice_score = 2 * jaccard / (jaccard + 1.0)

    true_labels = np.concatenate(gt)
    pred_scores = np.concatenate(pred)

    fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_scores)
    roc_auc = metrics.auc(fpr, tpr)
    eer, eer_threshold, eer_threshold_index = calculate_eer(fpr, tpr, thresholds)

    result_dict = {}

    result_dict["accuracy"] = accuracy
    result_dict["jaccard"] = jaccard
    result_dict["dice"] = dice_score
    result_dict["fpr"] = fpr
    result_dict["tpr"] = tpr
    result_dict["thresholds"] = thresholds
    result_dict["roc_auc"] = roc_auc
    result_dict["eer"] = eer
    result_dict["eer_threshold"] = eer_threshold
    result_dict["eer_threshold_index"] = eer_threshold_index

    save_path = os.path.join(model_dir, "finetuned_sam.pkl")

    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f)

    print("File saved: ", save_path)

    print("Accuracy: ", accuracy)
    print("Jaccard Index: ", jaccard)
    print("Dice Score: ", dice_score)

    print("False positive rate: ", fpr)
    print("True positive rate: ", tpr)
    print("Thresholds: ", thresholds)
    print("ROC-AUC: ", roc_auc)
    print("EER: ", eer)
    print("EER threshold: ", eer_threshold)
    print("EER threshold index: ", eer_threshold_index)




if __name__ == "__main__":
    test_model(data_directory=data_dir,
               model_dir=save_model_path,
               batch_size=batch_size,
               threshold=threshold)


