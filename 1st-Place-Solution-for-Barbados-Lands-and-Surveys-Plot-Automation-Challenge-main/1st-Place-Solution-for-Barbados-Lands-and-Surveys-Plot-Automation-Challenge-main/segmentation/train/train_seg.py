"""
Training pipeline for cadastral parcel segmentation with Lightning + SMP.

Trains Unet++ with EfficientNet-B5 encoder on 2048px images to predict binary masks
of land parcels. Uses K-fold cross-validation with heavy geometric augmentations
to handle scanning artifacts and geometric variations.

Input:
    - data/df.csv: unified dataset with image paths and pixel-space polygons
    - data/geom_px_df.csv: aligned polygon coordinates from VLM
    - configs/segmentation.yaml: training hyperparameters

Output:
    - outputs/seg_model_checkpoints/: trained Lightning checkpoints
"""

import os
import random
import sys
import warnings
from pathlib import Path

import __main__
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from box import Box
from PIL import Image, ImageDraw
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.transforms import ToTensor

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from utils.base_utils import load_config

BASE_CONFIG = load_config("configs/base.yaml")
SEG_CONFIG = load_config("configs/segmentation.yaml")
IMG_SIZE = SEG_CONFIG.image_size


ROOT_DIR = Path(BASE_CONFIG["root_dir"])
DATA_DIR = ROOT_DIR / "data"


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducible training across Python, NumPy, PyTorch, CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(seed, workers=True)


# Heavy augmentations: rotations, flips, perspective, noise, blur to mimic scan artifacts
train_transforms = A.Compose(
    [
        # Geometry (dominant)
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.25),
        A.OneOf(
            [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ],
            p=0.5,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.10,
            rotate_limit=10,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.75,
        ),
        A.Perspective(
            scale=(0.02, 0.04),
            keep_size=True,
            pad_mode=cv2.BORDER_CONSTANT,
            pad_val=0,
            mask_pad_val=0,
            p=0.10,
        ),
        A.OneOf(
            [
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0,
                ),
                # FIX: sigma >= 1
                A.ElasticTransform(
                    alpha=10,
                    sigma=4,  # <- was 0.7; must be >= 1
                    alpha_affine=10 * 0.03,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0,
                ),
            ],
            p=0.15,
        ),
        # Resize to canvas
        A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=1.0,
        ),
        # Light photometrics / scan artifacts
        A.OneOf(
            [
                A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, p=1.0),
            ],
            p=0.25,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.05, contrast_limit=0.05, p=1.0
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ],
            p=0.25,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.Downscale(scale_min=0.85, scale_max=0.95, p=1.0),
                A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
            ],
            p=0.20,
        ),
        A.Normalize(),
        ToTensorV2(),
    ]
)


# Validation transform: resize and pad only, no randomness
val_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=1.0,
        ),
        A.Normalize(),
        ToTensorV2(),
    ]
)


class PolyDataset(Dataset):
    """
    Load survey plan images and render polygon masks on-the-fly for segmentation training.

    Converts pixel-space polygon coordinates into binary masks and applies augmentations.
    """

    def __init__(
        self,
        df,
        transforms: Compose = None,
        mode: str = "train",
        image_key: str = "image_path",
        poly_key: str = "geometry_px",
        id_key: str = "ID",
    ):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode  # "train" | "val" | "test"
        self.image_key = image_key
        self.poly_key = poly_key
        self.id_key = id_key

    def __len__(self):
        return len(self.df)

    def _create_mask(self, poly, size):
        """Render polygon as binary mask using PIL."""
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        if poly and len(poly) >= 3:
            draw.polygon(poly, outline=1, fill=1)
        return np.array(mask, dtype=np.uint8)

    def __getitem__(self, idx):
        """Load image, render mask if available, apply transforms, return dict."""
        row = self.df.iloc[idx]
        image_path = Path(row[self.image_key])
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        # Only create mask if polygon exists and not test mode
        poly = None
        have_mask = (
            (self.mode != "test")
            and (self.poly_key in row)
            and isinstance(row[self.poly_key], (list, tuple))
        )
        if have_mask:
            poly = row[self.poly_key]

        img_np = np.array(img)
        if have_mask:
            mask_np = self._create_mask(poly, size=(img_w, img_h))
        else:
            mask_np = None

        if self.transforms:
            if mask_np is not None:
                augmented = self.transforms(image=img_np, mask=mask_np)
                img_tensor = augmented["image"]
                mask_tensor = augmented["mask"].unsqueeze(0).float()
            else:
                augmented = self.transforms(image=img_np)
                img_tensor = augmented["image"]
                mask_tensor = None
        else:
            img_tensor = ToTensor()(img)
            mask_tensor = (
                ToTensor()(Image.fromarray(mask_np)).float()
                if mask_np is not None
                else None
            )

        out = {
            "image": img_tensor,
            "image_path": str(image_path),
            "image_size": (img_w, img_h),
            "id": row[self.id_key],
        }
        if mask_tensor is not None:
            out["mask"] = mask_tensor
        return out


class PolyDataModule(pl.LightningDataModule):
    """
    Lightning data module providing train/val/test dataloaders.

    Wraps PolyDataset with configured batch sizes, workers, and transforms.
    """

    def __init__(
        self, train_df, val_df, train_transforms, val_transforms, cfg=None, test_df=None
    ):
        super().__init__()
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = None if test_df is None else test_df.reset_index(drop=True)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.cfg = cfg

    def train_dataloader(self):
        ds = PolyDataset(self.train_df, transforms=self.train_transforms, mode="train")
        return DataLoader(ds, **self.cfg.data_loader.train)

    def val_dataloader(self):
        ds = PolyDataset(self.val_df, transforms=self.val_transforms, mode="val")
        return DataLoader(ds, **self.cfg.data_loader.val)

    def predict_dataloader(self):
        if self.test_df is None:
            return None
        ds = PolyDataset(self.test_df, transforms=self.val_transforms, mode="test")
        # fall back to val loader params if test params not present
        dl_cfg = self.cfg.data_loader.val
        if hasattr(self.cfg.data_loader, "test"):
            dl_cfg = self.cfg.data_loader.test
        return DataLoader(ds, **dl_cfg)


class PolySegmenter(nn.Module):
    """
    Unet++ segmentation model with dropout regularization.

    Uses SMP's EfficientNet-B5 encoder with ImageNet pretraining.
    """

    def __init__(
        self,
        num_classes=1,
        in_chans=3,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        dropout_rate=0.3,
    ):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_chans,
            classes=num_classes,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x_img):
        x = self.model(x_img)
        if self.training:
            x = self.dropout(x)
        return x


class PolyLit(pl.LightningModule):
    """
    Lightning training module for polygon segmentation.

    Handles training loop, validation, optimizer (AdamW + CosineAnnealingLR),
    Jaccard loss, and IoU metrics.
    """

    def __init__(
        self,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_chans=3,
        lr=1e-3,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = PolySegmenter(
            num_classes=1,
            in_chans=in_chans,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            dropout_rate=dropout_rate,
        )

        self.lr = lr

        # Binary segmentation loss (IoU / Jaccard)
        self.criterion = smp.losses.JaccardLoss(mode="binary")

        # Metrics
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Forward pass, compute Jaccard loss, log metrics."""
        x = batch["image"]
        y = batch["mask"]  # [B, 1, H, W] or [B, H, W]
        logits = self(x)

        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_iou",
            self.train_iou(preds, y.int()),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation forward pass, log loss and IoU."""
        x = batch["image"]
        y = batch["mask"]
        logits = self(x)

        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_iou",
            self.val_iou(preds, y.int()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Inference forward pass returning binary masks."""
        x = batch["image"]
        logits = self(x)
        probs = torch.sigmoid(logits)
        masks = (probs > 0.5).float()
        return {
            "mask": masks.cpu(),
            "image_path": batch.get("image_path", None),
            "id": batch.get("id", None),
        }

    def configure_optimizers(self):
        """Setup AdamW optimizer with cosine annealing LR scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def main():
    """Load data, setup Lightning trainer, train segmentation model."""
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test = pd.read_csv(DATA_DIR / "Test.csv")
    ss = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    geom_px_df = pd.read_csv(DATA_DIR / "geom_px_df.csv")
    geom_px_df["geometry_px"] = geom_px_df["geometry_px"].apply(
        eval
    )  # Parse string to list
    df = pd.read_csv(DATA_DIR / "df.csv")
    df = df.drop(columns=["geometry_px"])
    df = df.merge(geom_px_df, on="ID", how="left")  # Merge pixel-space polygons
    df = df.dropna(subset=["image_path"], ignore_index=True)
    print(df.head(), df.shape)

    # Split by train/test, then fold for K-fold cross-validation
    train_data = df[df.split == "train"].reset_index(drop=True)
    train_df = train_data[train_data["fold"] != SEG_CONFIG.val_fold].reset_index(
        drop=True
    )
    val_df = train_data[train_data["fold"] == SEG_CONFIG.val_fold].reset_index(
        drop=True
    )
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Set deterministic seeds and enable TF32 for faster training
    seed_everything(SEG_CONFIG.seed)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    datamodule = PolyDataModule(
        train_df=train_df,
        val_df=val_df,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        cfg=SEG_CONFIG,
        test_df=test_df,
    )

    model = PolyLit(**SEG_CONFIG.model)

    # Setup callbacks: LR monitoring, early stopping, checkpointing
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval="step", log_momentum=False, log_weight_decay=False
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min"
    )

    EXP_NAME = SEG_CONFIG[f"exp_name"] + f"_fold{SEG_CONFIG.val_fold}"
    MODEL_SAVE_PATH = str(Path("outputs/seg_model_checkpoints")) + os.sep + EXP_NAME
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_SAVE_PATH,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        save_last=True,
        save_top_k=120,
        filename="model-{epoch}-{val_loss:.4f}-{val_iou:.4f}-{train_loss:.4f}-{train_iou:.4f}",
    )

    trainer = pl.Trainer(
        **SEG_CONFIG.trainer,
        callbacks=[model_checkpoint, lr_monitor, early_stop_callback],
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
