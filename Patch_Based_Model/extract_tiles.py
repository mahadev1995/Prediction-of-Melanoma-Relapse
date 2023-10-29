import cv2
import glob
import time
import numpy as np
import pandas as pd
from tifffile import imread
from skimage import color, morphology

# Define the path to the directory containing the TIFF images
path = '/home/janus/iwso-datasets/visiomel'
files = glob.glob(path + '/*.tif')
files.sort()
print('Number of Files: ', len(files))

# Load labels and annotations
labels = pd.read_csv('/home/janus/iwso-datasets/visiomel/train_labels.csv')
annotations = pd.read_csv('/home/janus/iwso-datasets/visiomel/annotations.csv')

print(f'Number of annotated images: {len(annotations)}')

def get_mask(image):
    """
    Generate a mask from an input image using color segmentation.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The generated mask.
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
        mode (int, optional): Padding mode for tile extraction.

    Returns:
        numpy.ndarray: Extracted tiles.
        bool: Indicates if enough tiles with information were extracted.

    source: https://github.com/drivendataorg/tissuenet-cervical-biopsies/blob/main/3rd%20Place/src/utils.py
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
    Concatenate tiles into a single image.

    Args:
        tiles (numpy.ndarray): Extracted tiles.
        n_tiles (int): The number of tiles.
        image_size (int): Size of the output image.

    Returns:
        numpy.ndarray: Concatenated image.
    
    source: https://github.com/drivendataorg/tissuenet-cervical-biopsies/blob/main/3rd%20Place/src/utils.py
    
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

# Process each image
for i in range(len(files)):
    tic = time.time()
    image_name = files[i].split('/')[-1].split('.')[0]
    label = labels[labels['filename'] == image_name + '.tif']['relapse'].values[0]

    image = imread(files[i], key=4)
    mask = get_mask(image)

    masked_image = mask[:, :, None] * image
    masked_image[masked_image == 0] = 255
    masked_image = np.clip(masked_image.astype(np.uint8), 0, 255)

    tiles, _ = get_tiles(image, 256, 36)
    stiched_image = concat_tiles(tiles, 36, 256)

    tiles, _ = get_tiles(masked_image, 256, 36)
    stiched_maskedimage = concat_tiles(tiles, 36, 256)
    
    np.save(f'./stiched_images/{label}/{image_name}.npy', stiched_maskedimage)

    toc = time.time()
    print(f'Saved tiles for {i+1} images, time_taken: {round((toc-tic)/60, 3)}')

    del image
    del tiles
    del masked_image
    del stiched_maskedimage
