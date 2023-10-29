import time
import random
import torch as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.dataset import MultimodalDataset
from model.model import MelanomaPredictor
from sklearn.metrics import (classification_report, 
                             roc_auc_score, 
                             f1_score, 
                             balanced_accuracy_score, 
                             accuracy_score, 
                             confusion_matrix
                            )

# Set random seed for reproducibility
random.seed(42)

# Define model and evaluation parameters
modelName = 'resnet34'
batch_size = 16
threshold = 0.5
size = 1024

# Define evaluation mode, method, checkpoint path, data paths, and output paths
mode = 'test'
method = 'two_stage_wt2'
ckpt_path = './checkpoints/resnet34_multimodal_2/checkpoint_22.ckp'
data_path = '/home/janus/iwso-datasets/visiomel'
test_label_path = f'../data/data_splits/{mode}/{mode}_labels.csv'

plot_name = f'confusion_matrix_ft_{mode}_{method}'
outpath_pred = f'./prediction/{method}/prediction_multimodal_{mode}_ckpt_39_{method}.npy'
outpath_gt = f'./prediction/{method}/ground_truth_{mode}_multimodal_{method}.npy'

# Load test dataset
test_dataset = MultimodalDataset(path=f'./data/{mode}_data.csv')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Initialize the MelanomaPredictor model
net = MelanomaPredictor(backbone_name=modelName, dropout=0.2).cuda()
net = t.nn.DataParallel(net)
ckp = t.load(ckpt_path)
net.load_state_dict(ckp['state_dict'])
net.eval()

print(f'Number of Batches: {len(test_loader)}')

predictions = []
ground_truth = []
batch_no = 0

# Perform predictions on the test dataset
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

# Apply threshold to predictions
y_pred = []
for i in predictions:
    if i > threshold:
        y_pred.append(1)
    else:
        y_pred.append(0)

# Evaluate the model
print(classification_report(ground_truth, y_pred))
print(roc_auc_score(ground_truth, predictions, average='macro'))
print(f1_score(ground_truth, y_pred, average='macro'))
print(accuracy_score(ground_truth, y_pred))
print(balanced_accuracy_score(ground_truth, y_pred))
cm = confusion_matrix(ground_truth, y_pred)

# Save predictions and ground truth
np.save(outpath_pred, predictions)
np.save(outpath_gt, ground_truth)

# Plot and save the confusion matrix
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(cm, annot=True, linewidths=0.01, ax=ax, cmap="Blues")
ax.set_xlabel('PREDICTED VALUES')
ax.set_ylabel('ACTUAL VALUES')
plt.savefig(f'./prediction/{method}/{plot_name}.png')
plt.savefig(f'./prediction/{method}/{plot_name}.svg')

# Plot and save the normalized confusion matrix
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(cmn, annot=True, linewidths=0.01, ax=ax, cmap="Blues")
ax.set_xlabel('PREDICTED VALUES')
ax.set_ylabel('ACTUAL VALUES')
plt.savefig(f'./prediction/{method}/{plot_name}_normalised.png')
plt.savefig(f'./prediction/{method}/{plot_name}_normalised.svg')
