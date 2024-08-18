# from backbones_unet.model.unet import Unet
from unet import Unet
# from backbones_unet.utils.dataset import SemanticSegmentationDataset
from dataset import SemanticSegmentationDataset
# from backbones_unet.utils.trainer import Trainer
from trainer import Trainer
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

csv_file = 'split_0.csv'
# image_dir = r'C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images'
image_dir = r'C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\normal_clean_mucosa\Normal clean mucosa'
image_dir_val = r'C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\foreign_body\Foreign body'

data = pd.read_csv(csv_file)
normal_data = data[data['label'] == 'Normal']
foreign_objects_data = data[data['label'] == 'Foreign Bodies']

# Split paths and labels
image_paths = [os.path.join(image_dir, f"{row['filename']}") for index, row in normal_data.iterrows()][:500]
mask_paths = [os.path.join(image_dir, f"{row['filename']}") for index, row in normal_data.iterrows()][:500]



image_paths_val = [os.path.join(image_dir_val, f"{row['filename']}") for index, row in foreign_objects_data.iterrows()][:2]
mask_paths_val = [os.path.join(image_dir_val, f"{row['filename']}") for index, row in foreign_objects_data.iterrows()][:2]




# train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
#     image_paths, mask_paths, test_size=0.2, random_state=42)


# create a torch.utils.data.Dataset/DataLoader
# train_img_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\normal_clean_mucosa"
# train_mask_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\normal_clean_mucosa"
#
# val_img_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\normal_clean_mucosa"
# val_mask_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\normal_clean_mucosa"
# train_img_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\overfittingsetfortesting"
# train_mask_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\overfittingsetfortesting"

# val_img_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\overfittingsetfortesting"
# val_mask_path = r"C:\Users\ben93\Downloads\kvasir-capsule-labeled-images\labelled_images\overfittingsetfortesting"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the inverse transformation for visualization
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])


def imshow(img):
    # Ensure the tensor is in the right format (C, H, W)
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)

    img = img.float()  # Convert to float to avoid issues in normalization

    if img.dim() == 2:  # If the image is grayscale, add a channel dimension
        img = img.unsqueeze(0)
    elif img.dim() == 3 and img.shape[2] == 3:  # If the image is (H, W, C)
        img = img.permute(2, 0, 1)  # Convert to (C, H, W)

    img = inverse_transform(img)  # Apply the inverse transform

    img = img.permute(1, 2, 0)  # Convert back to (H, W, C) for displaying
    img = img.numpy()
    img = np.clip(img, 0, 1)  # Ensure valid pixel range
    plt.imshow(img)
    # plt.show()


train_dataset = SemanticSegmentationDataset(image_paths, mask_paths,normalize=transform)
val_dataset = SemanticSegmentationDataset(image_paths_val, mask_paths_val,normalize=transform)

train_loader = DataLoader(train_dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1)

model = Unet(
    backbone='convnext_base', # backbone network name
    in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
    num_classes=3,            # output channels (number of classes in your dataset)
)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, 1e-4)
criterion = nn.MSELoss()
# criterion = nn.L1Loss


trainer = Trainer(
    model,                    # UNet model with pretrained backbone
    criterion=criterion,     # loss function for model convergence
    optimizer=optimizer,      # optimizer for regularization
    epochs=2                 # number of epochs for model training
)

trainer.fit(train_loader, val_loader)





# Function to calculate mean squared error
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# Function to test the model and visualize reconstruction quality
def test_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients needed
        for i, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            diff_imgs = abs(outputs-images)
            outputs = outputs.cpu().numpy()
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            diff_imgs = diff_imgs.cpu().numpy()

            for img, pred, gt, diff_img in zip(images, outputs, masks, diff_imgs):
                img = np.transpose(img, (1, 2, 0))
                pred = np.transpose(pred, (1, 2, 0))
                gt = np.transpose(gt, (1, 2, 0))
                diff_img = np.transpose(diff_img, (1, 2, 0))

                mse_value = mse(pred, gt)

                # Display images
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                imshow(img)
                plt.title('Original Image')
                plt.subplot(1, 3, 2)
                imshow(pred)
                plt.title(f'Predicted Mask - MSE: {mse_value:.2f}')
                plt.subplot(1, 3, 3)
                imshow(diff_img)
                plt.title('Reconstruction')
                plt.show()

            # Stop after displaying a few batches
            if i >= 2:  # display 3 batches
                break

# Testing the model
test_model(model, val_loader)
