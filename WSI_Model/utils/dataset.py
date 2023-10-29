import os
import cv2
import torch
import numpy as np
import pandas as pd
from tifffile import imread
from skimage import color, morphology
from torch.utils.data import Dataset

def get_mask(image):
    """
    Compute a mask for a given image using color manipulation and morphology operations.
    Args:
        image (numpy.ndarray): The input image.
    Returns:
        numpy.ndarray: The computed mask.
    """
    hed = color.rgb2hed(image)
    mask = morphology.remove_small_holes(hed[:, :, 1] > 0, 1000).astype(np.float32)

    _, counts = np.unique(mask, return_counts=True)
    if counts[1] / counts[0] < 0.11:
        dilation = cv2.dilate(mask, (9, 9), iterations=10)
        dilation = morphology.remove_small_holes(dilation > 0, 1000).astype(np.float32)
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(dilation, kernel, iterations=1)
        return eroded
    else:
        return mask

def get_tiles(img, tile_size, n_tiles, mode=0):
    """
    Extract tiles from an image.
    Args:
        img (numpy.ndarray): The input image.
        tile_size (int): The size of each tile.
        n_tiles (int): The number of tiles to extract.
        mode (int): Tile extraction mode (0: centered, 1: random).
    Returns:
        numpy.ndarray: Extracted tiles.
        bool: Whether enough tiles with useful information were extracted.
        # source: https://github.com/drivendataorg/tissuenet-cervical-biopsies/blob/main/3rd%20Place/src/utils.py 

    """
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img = img.reshape(
        img.shape[0] // tile_size, tile_size, img.shape[1] // tile_size, tile_size, 3
    )
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    n_tiles_with_info = (
        img.reshape(img.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255
    ).sum()
    if len(img) < n_tiles:
        img = np.pad(
            img, [[0, n_tiles - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255
        )

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:n_tiles]
    img = img[idxs]

    return img, n_tiles_with_info >= n_tiles

def concat_tiles(tiles, n_tiles, image_size):
    """
    Concatenate tiles to reconstruct the original image.

    Args:
        tiles (numpy.ndarray): Extracted tiles.
        n_tiles (int): The number of tiles.
        image_size (int): The size of the original image.

    Returns:
        numpy.ndarray: Reconstructed image.
        # source: https://github.com/drivendataorg/tissuenet-cervical-biopsies/blob/main/3rd%20Place/src/utils.py 

    """
    idxes = list(range(n_tiles))

    n_row_tiles = int(np.sqrt(n_tiles))
    img = np.zeros(
        (image_size * n_row_tiles, image_size * n_row_tiles, 3), dtype="uint8"
    )
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]
            else:
                this_img = np.ones((image_size, image_size, 3), dtype="uint8") * 255

            h1 = h * image_size
            w1 = w * image_size
            img[h1 : h1 + image_size, w1 : w1 + image_size] = this_img

    return img

class MelanomaDataset(Dataset):
    def __init__(self, labels, meta_data, meta_labels, data_path, transforms):
        """
        Custom PyTorch dataset for melanoma data.

        Args:
            labels (pandas.DataFrame): DataFrame containing labels.
            meta_data (pandas.DataFrame): DataFrame containing metadata.
            meta_labels (pandas.DataFrame): DataFrame containing meta-labels.
            data_path (str): Path to data directory.
            transforms: PyTorch data transformations to apply to images.
        """
        self.path = data_path
        self.files = np.array(labels)[:, 0]
        self.labels = np.array(labels)[:, 1]
        self.meta_data = meta_data
        self.meta_labels = meta_labels
        self.transforms = transforms

    def __len__(self):
        return len(list(self.labels))

    def get_image(self, image_path):
        """
        Load and return an image given its path
        Args:
            image_path (str): Path to the image file.
        Returns:
            numpy.ndarray: Loaded image.
        """
        image = imread(image_path, key=4)
        return image

    def get_metadata(self, filename):
        """
        Get metadata for a specific filename.
        Args:
            filename (str): Filename.
        Returns:
            numpy.ndarray: Metadata.
        """
        meta_data = self.meta_data[self.meta_data['filename'] == filename].values[:, 1:]
        return meta_data.astype(np.float32)

    def get_metalabels(self, filename):
        """
        Get meta-labels for a specific filename.
        Args:
            filename (str): Filename.
        Returns:
            torch.Tensor: Breslow thickness.
            torch.Tensor: Ulceration.
        """
        meta_labels = self.meta_labels[self.meta_labels['filename'] == filename]
        breslow = torch.tensor(meta_labels['breslow'].values)
        ulceration = torch.tensor(meta_labels['ulceration'].values)
        return breslow, ulceration

    def __getitem__(self, idx):
        filename = self.files[idx]
        relapse = self.labels[idx]
        relapse = torch.tensor(relapse)

        img_name = os.path.splitext(filename)[0] + '.npy'
        img_path = f'/home/vault/iwso/iwso089h/images/{relapse.item()}/{img_name}'
        image = np.load(img_path)

        if self.transforms:
            image = self.transforms(image)

        meta_data = self.get_metadata(filename)
        meta_data = torch.tensor(meta_data, dtype=torch.float32)

        breslow, ulceration = self.get_metalabels(filename)

        target = torch.zeros((3, ))
        target[0] = relapse
        target[1] = breslow
        target[2] = ulceration
        return image, meta_data, target

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset: Dataset, batch_size: int):
        """
        Custom sampler for creating balanced batches in a dataset.
        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): Batch size.
        """
        self.dataset = dataset
        self.labels = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
        self.positive_indices = []
        self.negative_indices = []
        for i, label in enumerate(self.labels):
            if label == 1:
                self.positive_indices.append(i)
            else:
                self.negative_indices.append(i)
        self.batch_size = batch_size

    def __iter__(self):
        positive_batches = [self.positive_indices[i:i + self.batch_size // 2] for i in range(0, len(self.positive_indices), self.batch_size // 2)]
        negative_batches = [self.negative_indices[i:i + self.batch_size // 2] for i in range(0, len(self.negative_indices), self.batch_size // 2)]
        batches = [None] * (len(positive_batches) + len(negative_batches))
        batches[::2] = positive_batches
        batches[1::2] = negative_batches
        for i in range(len(batches)):
            batch = batches[i]
            if len(batch) < self.batch_size:
                batch += batches[(i + 1) % len(batches)][:(self.batch_size - len(batch))]
            yield batch

    def __len__(self):
        return (len(self.positive_indices) + len(self.negative_indices)) // self.batch_size

class MultimodalDataset(Dataset):
    def __init__(self, path):
        """
        Custom PyTorch dataset for multimodal data.
        Args:
            path (str): Path to the dataset.
        """
        self.data = pd.read_csv(path)
        self.tabular = self.data[['filename', 'age', 'sex', 'body_site', 'history_yes', 'history_no']]
        self.target = self.data[['filename', 'breslow', 'ulceration', 'relapse']]
        self.image = self.data.drop(columns=['age', 'sex', 'body_site', 'history_yes', 'history_no'] + ['breslow', 'ulceration', 'relapse'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.loc[idx]['filename']

        tabular = self.tabular[self.tabular['filename'] == filename].values[:, 1:].astype(np.float32)
        image = self.image[self.image['filename'] == filename].values[:, 1:].astype(np.float32)

        targets = self.target[self.target['filename'] == filename]

        breslow = targets['breslow'].values[0]
        ulceration = targets['ulceration'].values[0]
        relapse = targets['relapse'].values[0]

        tabular = torch.from_numpy(tabular)
        image = torch.from_numpy(image)

        target = torch.zeros((3, ))
        target[0] = relapse
        target[1] = breslow
        target[2] = ulceration

        return image, tabular, target
