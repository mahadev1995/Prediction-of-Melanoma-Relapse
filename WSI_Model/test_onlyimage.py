# Import necessary libraries and modules
import os
import time
import random
import torch as t
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import MelanomaDataset
from model.image_only import ImageOnlyModel  # Import your model

# Set a random seed for reproducibility
random.seed(42)

# Define variables and paths
modelName = 'resnet34'
batch_size = 16
threshold = 0.5
size = 1024
data_path = '/home/janus/iwso-datasets/visiomel'
ckpt_path = '/home/woody/iwso/iwso089h/VisioMel/seminar/fullimage/checkpoints/resnet34_imageonly_bce_loss/checkpoint_48.ckp'
test_label_path = '../data/data_splits/val/val_labels.csv'

# Create an instance of the image-only model
net = ImageOnlyModel(backbone_name=modelName).cuda()
net = t.nn.DataParallel(net)

# Load the model checkpoint
ckp = t.load(ckpt_path)
net.load_state_dict(ckp['state_dict'])
net.eval();

# Load metadata and labels for testing
meta_data = pd.read_csv('../data/meta_data/meta_data_new.csv')
columns = ['filename', 'breslow', 'ulceration', 'relapse']
meta_labels = meta_data[columns]
meta_data = meta_data.drop(columns=columns[1:])
meta_data["age"] = meta_data["age"].div(100).round(2)
meta_data["body_site"] = meta_data["body_site"].div(14).round(2)
test_labels = pd.read_csv(test_label_path)

# Define image transformations
transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((size, size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

# Create a test dataset and dataloader
test_dataset = MelanomaDataset(test_labels, 
                                meta_data, 
                                meta_labels,
                                data_path, 
                                transforms=transform, 
                              )
test_loader= DataLoader(
                         test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False,
                         num_workers=8,
                         pin_memory=True
                         )

print(f'Number of Batches: {len(test_loader)}')

# Initialize lists to store predictions and ground truth
predictions = []
ground_truth = []
batch_no = 0

# Perform inference on the test dataset
with t.no_grad():
    for img, params, target in test_loader:
        tic = time.time()
        y_pred = net(img.cuda(), params.cuda())
        y_pred = y_pred[0].cpu().numpy()
        predictions.extend(list(y_pred))
        ground_truth.extend(list(target[:, 0].numpy()))
        toc = time.time()
        print(f"Complete prediction for batch: {batch_no}, time taken: {toc - tic}")
        batch_no += 1

# Apply a threshold to predictions to convert them to binary values (0 or 1)
y_pred = []
for i in predictions:
    if i > threshold:
        y_pred.append(1)
    else:
        y_pred.append(0)

# Print classification metrics
print(classification_report(ground_truth, y_pred))
print(roc_auc_score(ground_truth, predictions, average='macro'))
print(f1_score(ground_truth, y_pred, average='macro'))
print(accuracy_score(ground_truth, y_pred))
print(balanced_accuracy_score(ground_truth, y_pred))

# Compute and save the confusion matrix
cm = confusion_matrix(ground_truth, y_pred)
np.save('./prediction/prediction_onlyimage.npy', predictions)
np.save('./prediction/ground_truth.npy', ground_truth)

# Plot and save the confusion matrix
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(cm, annot=True, linewidths=0.01, ax=ax, cmap="Blues")
ax.set_xlabel('PREDICTED VALUES')
ax.set_ylabel('ACTUAL VALUES')
plt.savefig('./prediction/confusion_matrix.png')
plt.savefig('./prediction/confusion_matrix.svg')

# Compute and save the normalized confusion matrix
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(cmn, annot=True, linewidths=0.01, ax=ax, cmap="Blues")
ax.set_xlabel('PREDICTED VALUES')
ax.set_ylabel('ACTUAL VALUES')
plt.savefig('./prediction/confusion_matrix_normalised.png')
plt.savefig('./prediction/confusion_matrix_normalised.svg')
