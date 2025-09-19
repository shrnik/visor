import os
import re
import argparse
import h5py
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# -----------------------------
# Configurable parameters
# -----------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"  # or any Hugging Face vision model
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)

def check_if_embeddings_exist(hdf5_path: str) -> bool:
    return os.path.exists(hdf5_path)

def check_hdf5_file(hdf5_path: str, folder: str) -> bool:
    # len of embeddings should match number of images
    if not os.path.exists(hdf5_path):
        return False
    with h5py.File(hdf5_path, "r") as h5f:
        num_embeddings = len(h5f.keys())
    num_images = len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    return num_embeddings == num_images

def extract_datetime(folder: str, filename: str) -> str:
    """
    Combines the date from the folder name (e.g. 2025_09_01)
    with the time from the filename (e.g. 03_58_09.trig+00.jpg)
    into 'YYYY_MM_DD HH_MM_SS'.
    """
    date_part = os.path.basename(folder.rstrip("/"))
    match = re.search(r"(\d{2}_\d{2}_\d{2})", filename)
    time_part = match.group(1) if match else "00_00_00"
    return f"{date_part} {time_part}"

def get_hdf5_path(folder: str) -> str:
    return folder.rstrip("/") + "_embeddings.h5"
def process_folder(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    print(f"Found {len(files)} images in {folder}")
    hdf5_path = get_hdf5_path(folder)
    if check_if_embeddings_exist(hdf5_path) and check_hdf5_file(hdf5_path, folder):
        print(f"✅ Embeddings already exist and are valid at {hdf5_path}. Skipping...")
        return

    with h5py.File(hdf5_path, "w") as h5f:
        for i in tqdm(range(0, len(files), BATCH_SIZE), desc=f"Processing {folder}"):
            batch_files = files[i:i + BATCH_SIZE]
            images = [Image.open(os.path.join(folder, fname)).convert("RGB") for fname in batch_files]

            # Preprocess and move to GPU
            inputs = processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                outputs_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                embeddings = outputs_features.cpu().numpy()

            # Store each embedding with filename and full datetime stamp
            for fname, emb in zip(batch_files, embeddings):
                grp = h5f.create_group(fname)
                grp.create_dataset("embedding", data=emb.astype(np.float32), compression="gzip")
                grp.attrs["filename"] = fname
                grp.attrs["timestamp"] = extract_datetime(folder, fname)

    print(f"✅ Saved embeddings (with full date+time) to {hdf5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--test-hdf5", help="Test HDF5 file validation")
    args = parser.parse_args()
    
    if args.test_hdf5:
        hdf_path = get_hdf5_path(args.folder)
        if not os.path.exists(hdf_path):
            print(f"HDF5 file '{hdf_path}' does not exist.")
            exit(1)
        result = check_hdf5_file(hdf_path, args.folder)
        print(f"HDF5 file '{hdf_path}' is valid: {result}")
    else:
        process_folder(args.folder)
