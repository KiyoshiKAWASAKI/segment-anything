from datasets import load_dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from segdataset import SegmentationDataset
from torchvision import transforms
from transformers import SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
import sys, os
import warnings

warnings.filterwarnings('ignore')
from torch.nn.functional import threshold, normalize


# Define the smoothness loss function
def smoothness_loss(pred, lambda_smooth=0.1):
    # Compute gradients along x and y directions
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

    return lambda_smooth * (torch.mean(dx) + torch.mean(dy))


# Function to combine segmentation and smoothness loss
def combined_loss(pred, target, t, lambda_smooth=0.1):
    # Calculate segmentation loss
    seg_loss_value = seg_loss(pred, target)

    # Calculate smoothness loss
    smooth_loss_value = smoothness_loss(pred, lambda_smooth)

    # Combine the losses using the homotopy parameter t
    total_loss = (1 - t) * seg_loss_value + t * smooth_loss_value

    return total_loss


# Data and model directories
train_data_dir = "/project01/cvrl/jhuang24/australia-backup/data/test/"
# save_model_path = "/project01/cvrl/sabraha2/sam_data/finetune_v4_epochs_100"
# save_model_path = "/project01/cvrl/jhuang24/sam_data/sophia"
save_model_path = "/project01/cvrl/jhuang24/sam_data/sophia_50_epochs"

# Define customized dataset
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
data_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = SegmentationDataset(root=train_data_dir,
                                    processor=processor,
                                    image_folder='imgs',
                                    mask_folder='masks',
                                    transforms=data_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

batch = next(iter(train_dataloader))

# for k, v in batch.items():
#     print(k, v.shape)

print("GT mask: ", batch["ground_truth_mask"].shape)

# Load pretrain model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(num_epochs):
    epoch_losses = []
    t = epoch / num_epochs  # Calculate the homotopy parameter

    for batch in tqdm(train_dataloader):
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)

        # Calculate combined loss
        loss = combined_loss(predicted_masks, ground_truth_masks, t)

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    # Save model for each epoch
    model.save_pretrained(save_model_path)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean(epoch_losses):.4f}")

print("Training complete.")