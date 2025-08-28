import os
import numpy as np
import pandas as pd
from skimage.transform import resize

import matplotlib.pyplot as plt

# Parameters
roundup_cutoff = 0.5
input_files = {
    "train": "mnist_train.csv",
    "test": "mnist_test.csv"
}
output_dir = "../mnist12x12"
os.makedirs(output_dir, exist_ok=True)

def process_csv(filename, roundup_cutoff):
    df = pd.read_csv(filename)
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values.astype(np.float32) / 255.0  # scale to [0,1]
    # Downsize each image to 12x12
    downsized = []
    for img in images:
        img_2d = img.reshape(28, 28)
        img_small = resize(img_2d, (12, 12), order=1, anti_aliasing=True, preserve_range=True)
        img_bin = (img_small >= roundup_cutoff).astype(np.uint8)
        downsized.append(img_bin)
    downsized = np.array(downsized)
    return downsized, labels

# Process and save
for split in ["train", "test"]:
    feats, labels = process_csv(input_files[split], roundup_cutoff)
    np.save(os.path.join(output_dir, f"mnist12x12_{split}feats.npy"), feats)
    np.save(os.path.join(output_dir, f"mnist12x12_{split}labels.npy"), labels)

# Plot 16 random images from train set
feats = np.load(os.path.join(output_dir, "mnist12x12_trainfeats.npy"))
labels = np.load(os.path.join(output_dir, "mnist12x12_trainlabels.npy"))
idxs = np.random.choice(feats.shape[0], 16, replace=False)
plt.figure(figsize=(8, 8))
for i, idx in enumerate(idxs):
    plt.subplot(4, 4, i+1)
    plt.imshow(feats[idx], cmap='gray', vmin=0, vmax=1)
    plt.title(str(labels[idx]))
    plt.axis('off')
plt.tight_layout()
plt.show()