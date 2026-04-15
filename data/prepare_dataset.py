"""Script to recreate the PolypGen6 datasets."""

import ast
import os
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from synapseclient import Synapse


def download_polypgen(auth_token:str, target_dir:str) -> None:
    """Download the PolypGen dataset from Synapse."""
    syn = Synapse()
    syn.login(authToken=auth_token)
    syn.get("syn45200214", downloadLocation=target_dir)
    # Unzip 
    with zipfile.ZipFile(f"{target_dir}/polypgen2021_multicenterdata_v3.zip", 'r') as z:
        z.extractall(target_dir)
    os.remove(f"{target_dir}/polypgen2021_multicenterdata_v3.zip")


def get_sequence_data(target_dir:str) -> None:
    """Keep only the sequence data folder and remove the rest."""
    sequence_data_path = os.path.join(f"{target_dir}/PolypGen2021_MultiCenterData_v3/sequenceData")
    shutil.move(sequence_data_path, f"{target_dir}/sequenceData")
    shutil.rmtree(f"{target_dir}/__MACOSX")
    shutil.rmtree(f"{target_dir}/PolypGen2021_MultiCenterData_v3")


def parse_cropbox(cropbox_str:str) -> tuple:
    """
    Convert a cropbox string like "[690,4,1920-160,1080-6]"
    into a tuple of integers (690, 4, 1760, 1074).
    """
    cropbox_str = cropbox_str.strip("[]")
    parts = cropbox_str.split(",")
    evaluated = []
    for p in parts:
        p = p.strip()
        if "-" in p:
            left, right = p.split("-")
            evaluated.append(int(left) - int(right))
        else:
            evaluated.append(int(p))
    return tuple(evaluated)


def find_frame(frame_id:int, folder:str, get_mask:bool=False) -> str:
    """Get the name of the frame's image file."""
    if get_mask:
        pattern = re.compile(rf".*_0*{frame_id}_mask\.jpg$", re.IGNORECASE)
    else:
        pattern = re.compile(rf".*_0*{frame_id}\.jpg$", re.IGNORECASE)

    matches = [f for f in os.listdir(folder) if pattern.match(f)]

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for frame {frame_id} in {folder}")

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for frame {frame_id} in {folder}: {matches}"
        )

    filename = matches[0]
    return filename


def preprocess_frame(frame_dir:str, crop_box:tuple[int, int, int, int], result_dir:str) -> None:
    """Crop the frame using its assigned cropbox, and resize it to 224x224 pixels."""
    with Image.open(frame_dir) as img:
        # Crop to 1:1 aspect ratio
        cropped = img.crop(crop_box)
        w, h = cropped.size
        if w != h:
            raise ValueError(f"Cropped size not square:{cropped.size}")
        # Resize to 224x224 px
        img_224 = cropped.resize((224, 224), Image.BILINEAR)
        # Save file
        img_224.save(result_dir)


def extract_data_samples(csv_dir:str, raw_dir:str) -> None:
    """Get the PolypGen6 data samples."""
    df = pd.read_csv(csv_dir, sep=";")

    for idx, row in df.iterrows():
        split = row["dataset"]
        video_id = row["video_id"]
        source_folder = row["sourceFolder"].replace("\\", "/")
        label = int(row["label"])
        cropbox = parse_cropbox(row["cropbox"])
        frame_ids = ast.literal_eval(row["id_frames"])

        original_video_id = os.path.basename(source_folder)

        data_result_dir = f"./data/polypGen6/data/{split}/{video_id}"
        os.makedirs(data_result_dir, exist_ok=True)

        mask_result_dir = f"./data/polypGen6/mask/{split}/{video_id}"
        os.makedirs(mask_result_dir, exist_ok=True)
    
        if (label == 1) or (original_video_id in ["seq1", "seq7"]) or (video_id in ["seq17_clip1", "seq17_clip2"]):
            for idx, frame_id in enumerate(frame_ids, start=1):
                # Extract video frame, crop and resize it to 224x224 pixels
                data_dir = f"{raw_dir}/{source_folder}/images_{original_video_id}"
                filename = find_frame(frame_id, data_dir, get_mask=False)
                preprocess_frame(
                    frame_dir=f"{data_dir}/{filename}", 
                    crop_box=cropbox, 
                    result_dir=f"{data_result_dir}/{idx}.jpg",
                )
                # Extract segmentation mask, crop and resize it to 224x224 pixels
                mask_dir = f"{raw_dir}/{source_folder}/masks_{original_video_id}"
                filename = find_frame(frame_id, mask_dir, get_mask=True)
                preprocess_frame(
                    frame_dir=f"{mask_dir}/{filename}", 
                    crop_box=cropbox, 
                    result_dir=f"{mask_result_dir}/{idx}.jpg",
                )

        else:
            for idx, frame_id in enumerate(frame_ids, start=1):
                # Extract video frames
                data_dir = f"{raw_dir}/{source_folder}"
                filename = find_frame(frame_id, data_dir, get_mask=False)
                preprocess_frame(
                    frame_dir=f"{data_dir}/{filename}", 
                    crop_box=cropbox, 
                    result_dir=f"{data_result_dir}/{idx}.jpg",
                )
                # Create empty segmentation masks
                mask = Image.new("L", (224, 224), 0)
                mask.save(f"{mask_result_dir}/{idx}.jpg")
        
    # Remove the raw data
    shutil.rmtree(raw_dir)


def get_auth_token() -> str:
    """Load the Synapse token from the local environment."""
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    auth_token = os.getenv("SYNAPSE_AUTH_TOKEN", "").strip()
    if not auth_token:
        raise ValueError(
            "Missing SYNAPSE_AUTH_TOKEN. Copy '.env.example' to '.env', "
            "set SYNAPSE_AUTH_TOKEN to your Synapse Personal Access Token, "
            "then rerun 'python -m data.prepare_dataset'."
        )

    return auth_token


if __name__ == "__main__":
    auth_token = get_auth_token()

    raw_dir = "./data/polypgen"
    download_polypgen(auth_token, raw_dir)
    get_sequence_data(raw_dir)

    csv_dir = "./data/polypGen6/meta_data.csv"
    extract_data_samples(csv_dir, raw_dir)
