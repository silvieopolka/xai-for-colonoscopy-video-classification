"""Dataset utilities for the PolypGen clip data."""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from typing import Callable, Optional, Sequence, Tuple


def collate_polypgen_batch(batch:list):
    """Custom collate function for PolypGenDataset that handles (clip, label, video_id, masks) tuples.
    
    Args:
        batch: List of (clip, label, video_id, masks) tuples from the dataset
        
    Returns:
        Tuple of (clips_tensor, labels_tensor, video_ids_list, masks_list)
        where masks_list contains numpy arrays or None values
    """
    clips = []
    labels = []
    video_ids = []
    masks_list = []
    for clip, label, video_id, masks in batch:
        clips.append(clip)
        labels.append(label)
        video_ids.append(video_id)
        masks_list.append(masks)
    # Stack clips and labels into tensors
    clips_tensor = torch.stack(clips, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return clips_tensor, labels_tensor, video_ids, masks_list


class PolypGenDataset(Dataset):
    """PyTorch Dataset for pre-sliced PolypGen clips."""

    # standardize with ImageNet mean/std values (cached to avoid per-call allocs)
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(
        self,
        split: str,
        num_frames: int = 6,
        frame_indices: Optional[Sequence[int]] = None,
        load_masks: bool = False,
        data_root: str = "./data/polypGen6/data",
        masks_root: str = "./data/polypGen6/mask",
        meta_path: str = "./data/polypGen6/meta_data.csv",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        temporal_reverse_prob: float = 0.0,
        run_sanity_check: bool = True,
        sanity_check_samples: int = 5,
        sanity_check_seed: Optional[int] = None,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> None:
        
        if labels_df is None and split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Use 'train', 'val', or 'test'.")

        if not 0.0 <= temporal_reverse_prob <= 1.0:
            raise ValueError(f"'temporal_reverse_prob' must be in [0.0, 1.0], got {temporal_reverse_prob}.")

        self.split = split if labels_df is None else "cross_validation"

        if frame_indices is None:
            if num_frames <= 0:
                raise ValueError(f"num_frames must be > 0, got {num_frames}.")
            # Choose the first num_frames by default
            self.frame_indices = list(range(1, num_frames + 1))
            self.num_frames = num_frames
        else:
            if len(frame_indices) == 0:
                raise ValueError("frame_indices must not be empty.")
            self.frame_indices = [int(i) for i in frame_indices]
            if num_frames != len(self.frame_indices):
                raise ValueError(f"num_frames ({num_frames}) must match len(frame_indices) ({len(self.frame_indices)}).")
            if any(i < 1 for i in self.frame_indices):
                raise ValueError(f"All frame indices must be >= 1, got {self.frame_indices}.")
            self.num_frames = len(self.frame_indices)

        self.load_masks = load_masks
        self.data_root = data_root
        self.masks_root = masks_root
        self.meta_path = meta_path
        self.transform = transform or ClipConsistentTransform(train=False)
        self.temporal_reverse_prob = temporal_reverse_prob
        self.run_sanity_check = run_sanity_check
        self._sanity_rng = random.Random(sanity_check_seed)  # allow deterministic sampling

        # Load metadata
        if labels_df is None:
            df = pd.read_csv(meta_path, sep=";")
        else:
            df = labels_df.copy()
        required_cols = {"dataset", "video_id", "label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {meta_path}.")

        if labels_df is None:
            # Get the data samples for the requested split
            self.labels = df[df["dataset"] == split].reset_index(drop=True)  # enforce split integrity
        else:
            # Get the prepared data samples (cross-validation)
            self.labels = df.reset_index(drop=True)

        if len(self.labels) == 0:
            msg = f"No samples found for split '{split}' in {meta_path}." if labels_df is None else "No samples found in provided labels_df."
            raise ValueError(msg)

        if run_sanity_check:
            self._sanity_check(sample_size=min(sanity_check_samples, len(self.labels)))


    def __len__(self) -> int:
        """Get the number of input samples in the dataset."""
        return len(self.labels)


    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, int, str, Optional[np.ndarray]]:
        """Get a specific input sample of the dataset."""
        row = self.labels.loc[idx]
        video_id = row["video_id"]
        label = int(row["label"])
        video_ds = row["dataset"]

        frame_paths = self._frame_paths_for_video(str(video_id), ds=video_ds)

        # Check whether data sample contains the required amount of frames
        missing = self._missing_frames(frame_paths) 
        if missing:
            raise FileNotFoundError(
                f"Missing frames {missing} for clip '{video_id}' in split '{self.split}'. "
                f"Expected frame indices {self.frame_indices}."
            )

        # Randomly choose the data augmentations for the clip (if any)
        if hasattr(self.transform, "start_new_clip"):
            self.transform.start_new_clip()
        
        # Apply the same augmentations to each frame in the clip
        frames = []
        for p in frame_paths:
            with Image.open(p) as img:
                frames.append(self.transform(img.convert("RGB")))
        clip = torch.stack(frames, dim=1)  # final shape: (C, T, H, W)

        # Optionally apply temporal reversal of the video as a video-level augmentation
        if self.temporal_reverse_prob > 0.0 and random.random() < self.temporal_reverse_prob:
            clip = torch.flip(clip, dims=[1])
        
        # Load masks if requested
        masks = None
        if self.load_masks:
            masks = self._load_masks_for_clip(video_id, ds=video_ds)
        return clip, label, str(video_id), masks


    def _sanity_check(self, sample_size:int) -> None:
        """Verify a sample of clips contains the expected numbered frames to fail fast if raw data is present."""
        sample_indices = self._sanity_rng.sample(range(len(self.labels)), k=sample_size)
        for idx in sample_indices:
            row = self.labels.loc[idx]
            video_id = row["video_id"]
            video_ds = row["dataset"]
            frame_paths = self._frame_paths_for_video(str(video_id), ds=video_ds)
            missing = self._missing_frames(frame_paths)
            if missing:
                raise FileNotFoundError(
                    f"Sanity check failed: missing frames {missing} for clip '{video_id}' in split '{self.split}'. "
                    f"If you only have raw videos, run slicing first. Expected frame indices {self.frame_indices}."
                )


    def _frame_paths_for_video(self, video_id:str, ds:str):
        """Get all paths to the requested frame .jpg files."""
        video_dir = os.path.join(self.data_root, ds, video_id)
        return [os.path.join(video_dir, f"{i}.jpg") for i in self.frame_indices]

    @staticmethod
    def _missing_frames(frame_paths:list[str]):
        """Get the missing frames. Returns an empty list when no frames are missing."""
        return [os.path.basename(p) for p in frame_paths if not os.path.isfile(p)]


    def _load_masks_for_clip(self, video_id:str, ds:str) -> Optional[np.ndarray]:
        """Load masks for all frames in a clip.
        
            Args:
                video_id: The video clip identifier (e.g., 'seq1_clip0').
                ds: The original dataset of the video clip ('train', 'val' or 'test').
            
            Returns:
                A numpy array of shape (T, H, W) containing binary masks, or None if not found
        """
        mask_dir = Path(self.masks_root) / str(ds) / str(video_id)
        
        if not mask_dir.exists():
            return None
        
        # Load masks for each frame
        mask_list = []
        for frame_num in range(1, self.num_frames + 1):
            mask_path = mask_dir / f"{frame_num}.jpg"
            try:
                mask_img = Image.open(mask_path).convert("L")
                # Check if mask has the same dimensions as the video frame
                if mask_img.size != (224, 224):
                    raise ValueError(f"Mask {mask_path} has size {mask_img.size}, expected (224, 224).")
                # Convert to binary array
                mask_array = np.array(mask_img)
                mask_array = (mask_array > 127).astype(np.uint8)
                mask_list.append(mask_array)
            except Exception as e:
                # If loading fails, use empty mask
                mask_list.append(np.zeros((224, 224), dtype=np.uint8))
        
        if mask_list:
            return np.stack(mask_list, axis=0)   # shape: (T, H, W)
        else:
            return None


    @staticmethod
    def _default_transform(img:Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor using ImageNet mean/std."""
        arr = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0
        return (arr - PolypGenDataset._IMAGENET_MEAN) / PolypGenDataset._IMAGENET_STD


class ClipConsistentTransform:
    """Augmentations sampled once per clip to keep frames aligned.
    
    All augmentation parameters can be configured individually.
    Set probability to 0.0 to disable an augmentation.
    """

    def __init__(
        self, 
        train: bool = True, 
        # Flip augmentations
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,
        # Color jitter
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
        saturation_jitter: float = 0.2,
        hue_jitter: float = 0.0,
        # Blur augmentations
        gaussian_blur_prob: float = 0.5,
        gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0),
        motion_blur_prob: float = 0.0,
        motion_blur_kernel_size: int = 5,
        # Geometric augmentations
        rotation_prob: float = 0.0,
        rotation_degrees: float = 15.0,
        random_crop_prob: float = 0.0,
        random_crop_scale: Tuple[float, float] = (0.85, 1.0),
        perspective_prob: float = 0.0,
        perspective_distortion: float = 0.1,
        # Noise augmentations
        gaussian_noise_prob: float = 0.0,
        gaussian_noise_std: float = 0.05,
    ):

        self.train = train

        # Validate probabilities
        for name, prob in [
            ('horizontal_flip_prob', horizontal_flip_prob),
            ('vertical_flip_prob', vertical_flip_prob),
            ('gaussian_blur_prob', gaussian_blur_prob),
            ('motion_blur_prob', motion_blur_prob),
            ('rotation_prob', rotation_prob),
            ('random_crop_prob', random_crop_prob),
            ('perspective_prob', perspective_prob),
            ('gaussian_noise_prob', gaussian_noise_prob),
        ]:
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got{prob}.")
        
        # Store configuration
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.saturation_jitter = saturation_jitter
        self.hue_jitter = hue_jitter
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel_size = motion_blur_kernel_size
        self.rotation_prob = rotation_prob
        self.rotation_degrees = rotation_degrees
        self.random_crop_prob = random_crop_prob
        self.random_crop_scale = random_crop_scale
        self.perspective_prob = perspective_prob
        self.perspective_distortion = perspective_distortion
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        
        self._set_noop_params()


    def _set_noop_params(self):
        """Set all augmentation parameters to no-op values."""
        self.do_hflip = False
        self.do_vflip = False
        self.b_factor = 1.0
        self.c_factor = 1.0
        self.s_factor = 1.0
        self.h_factor = 0.0
        self.gaussian_blur_sigma_val = None
        self.do_motion_blur = False
        self.rotation_angle = 0.0
        self.do_random_crop = False
        self.crop_params = None
        self.do_perspective = False
        self.perspective_params = None
        self.do_gaussian_noise = False
        self.noise_std = 0.0


    def start_new_clip(self):
        """Sample augmentation parameters for a new clip."""
        if not self.train:
            self._set_noop_params()
            return
        
        # Flip augmentations
        self.do_hflip = random.random() < self.horizontal_flip_prob
        self.do_vflip = random.random() < self.vertical_flip_prob
        
        # Color jitter
        self.b_factor = random.uniform(1 - self.brightness_jitter, 1 + self.brightness_jitter) if self.brightness_jitter > 0 else 1.0
        self.c_factor = random.uniform(1 - self.contrast_jitter, 1 + self.contrast_jitter) if self.contrast_jitter > 0 else 1.0
        self.s_factor = random.uniform(1 - self.saturation_jitter, 1 + self.saturation_jitter) if self.saturation_jitter > 0 else 1.0
        self.h_factor = random.uniform(-self.hue_jitter, self.hue_jitter) if self.hue_jitter > 0 else 0.0
        
        # Gaussian blur
        if random.random() < self.gaussian_blur_prob:
            self.gaussian_blur_sigma_val = random.uniform(*self.gaussian_blur_sigma)
        else:
            self.gaussian_blur_sigma_val = None
        
        # Motion blur
        self.do_motion_blur = random.random() < self.motion_blur_prob
        
        # Rotation
        if random.random() < self.rotation_prob:
            self.rotation_angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        else:
            self.rotation_angle = 0.0
        
        # Random crop
        self.do_random_crop = random.random() < self.random_crop_prob
        if self.do_random_crop:
            scale = random.uniform(*self.random_crop_scale)
            ratio = random.uniform(0.95, 1.05)  # Near-square aspect ratio
            self.crop_params = (scale, ratio)
        else:
            self.crop_params = None
        
        # Perspective transform
        self.do_perspective = random.random() < self.perspective_prob
        if self.do_perspective:
            # Generate random perspective distortion points
            distortion = self.perspective_distortion
            self.perspective_params = distortion
        else:
            self.perspective_params = None
        
        # Gaussian noise
        self.do_gaussian_noise = random.random() < self.gaussian_noise_prob
        if self.do_gaussian_noise:
            self.noise_std = self.gaussian_noise_std
        else:
            self.noise_std = 0.0


    def __call__(self, img:Image.Image) -> torch.Tensor:
        """Apply augmentations to a single frame."""
        # Random crop (applied first to get the region)
        if self.do_random_crop and self.crop_params is not None:
            scale, ratio = self.crop_params
            width, height = img.size
            area = width * height
            target_area = area * scale
            w = int(round((target_area * ratio) ** 0.5))
            h = int(round((target_area / ratio) ** 0.5))
            
            # Ensure dimensions are valid
            w = min(w, width)
            h = min(h, height)
            
            # Random position
            i = random.randint(0, height - h) if height > h else 0
            j = random.randint(0, width - w) if width > w else 0
            
            img = F.crop(img, i, j, h, w)
            img = F.resize(img, (224, 224))
        else:
            # Always resize to 224x224 if no random crop is applied
            img = F.resize(img, (224, 224))
        
        # Rotation
        if self.rotation_angle != 0.0:
            img = F.rotate(img, self.rotation_angle, interpolation=F.InterpolationMode.BILINEAR)
        
        # Perspective transform
        if self.do_perspective and self.perspective_params is not None:
            width, height = img.size
            distortion = self.perspective_params
            
            # Define startpoints (corners of the image)
            startpoints = [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ]
            
            # Apply random distortion to endpoints
            endpoints = [
                [
                    max(0, min(width - 1, x + random.uniform(-distortion * width, distortion * width))),
                    max(0, min(height - 1, y + random.uniform(-distortion * height, distortion * height)))
                ]
                for x, y in startpoints
            ]
            
            img = F.perspective(img, startpoints, endpoints, interpolation=F.InterpolationMode.BILINEAR)
        
        # Horizontal flip
        if self.do_hflip:
            img = F.hflip(img)
        
        # Vertical flip
        if self.do_vflip:
            img = F.vflip(img)
        
        # Color jitter
        if self.b_factor != 1.0:
            img = F.adjust_brightness(img, self.b_factor)
        if self.c_factor != 1.0:
            img = F.adjust_contrast(img, self.c_factor)
        if self.s_factor != 1.0:
            img = F.adjust_saturation(img, self.s_factor)
        if self.h_factor != 0.0:
            img = F.adjust_hue(img, self.h_factor)
        
        # Gaussian blur
        if self.gaussian_blur_sigma_val is not None:
            img = F.gaussian_blur(img, kernel_size=3, sigma=self.gaussian_blur_sigma_val)
        
        # Motion blur (directional blur)
        if self.do_motion_blur:
            # Create motion blur kernel (random direction)
            angle = random.uniform(0, 360)
            kernel_size = self.motion_blur_kernel_size
            img = img.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))
        
        # Convert to tensor
        arr = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0
        
        # Add Gaussian noise (in tensor space)
        if self.do_gaussian_noise and self.noise_std > 0:
            noise = torch.randn_like(arr) * self.noise_std
            arr = arr + noise
            arr = torch.clamp(arr, 0.0, 1.0)
        
        # Normalize with ImageNet stats
        return (arr - PolypGenDataset._IMAGENET_MEAN) / PolypGenDataset._IMAGENET_STD


def load_dataset_split(dataset:str, split:str, load_masks:bool=False) -> PolypGenDataset:
        """Load a specific split of a PolypGen dataset (without data augmentation)."""
        if dataset == "polypGen6":
            num_frames = 6
        else:
            raise ValueError(f"'{dataset}' is an invalid dataset; use 'polypGen6'.")
        ds = PolypGenDataset(
            split=split,
            num_frames=num_frames,
            load_masks=load_masks,
            data_root=f"./data/{dataset}/data",
            masks_root=f"./data/{dataset}/mask",
            meta_path=f"./data/{dataset}/meta_data.csv",
        )
        return ds


def get_polypgen_datasets(
    num_frames: int = 6,
    frame_indices: Optional[Sequence[int]] = None,
    load_masks: bool = False,
    data_root: str = "./data/polypGen6/data",
    masks_root: str = "./data/polypGen6/mask",
    meta_path: str = "./data/polypGen6/meta_data.csv",
    run_sanity_check: bool = True,
    sanity_check_samples: int = 5,
    sanity_check_seed: Optional[int] = 42,
    train_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    eval_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    # Temporal augmentation (applied at clip level)
    temporal_reverse_prob: float = 0.0,
    # Augmentation configuration (applied at frame level)
    aug_config: Optional[dict] = None,
    # Cross Validation
    apply_cross_validation: bool = False,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
):
    """Factory to build PolypGen6 train/val/test datasets with shared defaults.
    
        Args:
            aug_config: Dictionary of augmentation parameters to pass to ClipConsistentTransform.
                        If None, uses default configuration with backward compatibility.
    """
    # Backward compatibility: if aug_config not provided, use defaults
    if aug_config is None:
        aug_config = {}

    # Default transforms: clip-consistent aug for train, identity+norm for eval
    train_transform = train_transform or ClipConsistentTransform(train=True, **aug_config)
    eval_transform = eval_transform or ClipConsistentTransform(train=False)

    # ---------------------------------------------------------------------------------------
    # K-FOLD CROSS VALIDATION
    # ---------------------------------------------------------------------------------------
    if apply_cross_validation:
        # Load full metadata and compute stratified folds over train+val
        df = pd.read_csv(meta_path, sep=';')
        required_cols = {"dataset", "video_id", "label", "sourceFolder"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {meta_path}.")
        
        train_val_df = df[df["dataset"].isin({"train", "val"})].reset_index(drop=True)
        if len(train_val_df) == 0:
            raise ValueError("No samples found in 'train' or 'val' splits for cross-validation.")

        labels = train_val_df["label"].astype(int).to_numpy()
        orig_clip_groups = train_val_df["sourceFolder"].astype(str).to_numpy()
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        # Get the train and validation datasets for each fold (ensuring subclips of the same original clip are never split across folds)
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_df, labels, orig_clip_groups)):
            print(f"Fold {fold_idx}:")
            print(f"  Train groups:    {np.unique(np.array(orig_clip_groups[train_idx].tolist()))}")
            print(f"  Val groups:      {np.unique(np.array(orig_clip_groups[val_idx].tolist()))}")
            
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

            train_ds = PolypGenDataset(
                split="train",
                num_frames=num_frames,
                frame_indices=frame_indices,
                load_masks=load_masks,
                data_root=data_root,
                masks_root=masks_root,
                meta_path=meta_path,
                transform=train_transform,
                temporal_reverse_prob=temporal_reverse_prob,
                run_sanity_check=run_sanity_check,
                sanity_check_samples=sanity_check_samples,
                sanity_check_seed=sanity_check_seed,
                labels_df=train_df,
            )

            val_ds = PolypGenDataset(
                split="val",
                num_frames=num_frames,
                frame_indices=frame_indices,
                load_masks=load_masks,
                data_root=data_root,
                masks_root=masks_root,
                meta_path=meta_path,
                transform=eval_transform,
                temporal_reverse_prob=0.0,
                run_sanity_check=run_sanity_check,
                sanity_check_samples=sanity_check_samples,
                sanity_check_seed=sanity_check_seed,
                labels_df=val_df,
            )

            folds.append((train_ds, val_ds))
        
        # Get the fixed test dataset
        test_ds = PolypGenDataset(
            split="test",
            num_frames=num_frames,
            frame_indices=frame_indices,
            load_masks=load_masks,
            data_root=data_root,
            masks_root=masks_root,
            meta_path=meta_path,
            transform=eval_transform,
            temporal_reverse_prob=0.0,
            run_sanity_check=run_sanity_check,
            sanity_check_samples=sanity_check_samples,
            sanity_check_seed=sanity_check_seed,
        )

        return (folds, test_ds)
    
    # ---------------------------------------------------------------------------------------
    # SINGLE DATASET DIVISION 
    # Note: The val data includes more difficult examples compared to train and test datasets.
    # ---------------------------------------------------------------------------------------
    else: 
        train_ds = PolypGenDataset(
            split="train",
            num_frames=num_frames,
            frame_indices=frame_indices,
            load_masks=load_masks,
            data_root=data_root,
            masks_root=masks_root,
            meta_path=meta_path,
            transform=train_transform,
            temporal_reverse_prob=temporal_reverse_prob,
            run_sanity_check=run_sanity_check,
            sanity_check_samples=sanity_check_samples,
            sanity_check_seed=sanity_check_seed,
        )

        val_ds = PolypGenDataset(
            split="val",
            num_frames=num_frames,
            frame_indices=frame_indices,
            load_masks=load_masks,
            data_root=data_root,
            masks_root=masks_root,
            meta_path=meta_path,
            transform=eval_transform,
            temporal_reverse_prob=0.0,
            run_sanity_check=run_sanity_check,
            sanity_check_samples=sanity_check_samples,
            sanity_check_seed=sanity_check_seed,
        )

        test_ds = PolypGenDataset(
            split="test",
            num_frames=num_frames,
            frame_indices=frame_indices,
            load_masks=load_masks,
            data_root=data_root,
            masks_root=masks_root,
            meta_path=meta_path,
            transform=eval_transform,
            temporal_reverse_prob=0.0,
            run_sanity_check=run_sanity_check,
            sanity_check_samples=sanity_check_samples,
            sanity_check_seed=sanity_check_seed,
        )

        return (train_ds, val_ds, test_ds)