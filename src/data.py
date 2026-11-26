from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_ham10000(data_dir, split_csv):
    """Read the saved split and attach the path of each image."""
    df = pd.read_csv(split_csv)
    paths = {p.stem: p for p in Path(data_dir).rglob("*.jpg")}
    df["path"] = df["image_id"].map(paths)
    assert df["path"].notna().all(), "some images were not found"
    return df


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SkinDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])

def load_pad_ufes_20(data_dir):
    """Read PAD-UFES-20 (smartphone images) and make it look like our HAM10000 table."""
    data_dir = Path(data_dir)
    meta_file = next(data_dir.rglob("metadata.csv"))              # find the table, in any subfolder
    df = pd.read_csv(meta_file)

    paths = {p.stem: p for p in data_dir.rglob("*.png")}
    df["image_id"] = df["img_id"].astype(str).map(lambda s: Path(s).stem)   # works with or without ".png"
    df["path"] = df["image_id"].map(paths)

    # Keep only the four classes that also exist in HAM10000
    #   MEL, BCC = malignant (1); NEV, SEK = benign (0)
    df = df[df["diagnostic"].isin(["MEL", "BCC", "NEV", "SEK"])].copy()
    df["label"] = df["diagnostic"].isin(["MEL", "BCC"]).astype(int)
    df["dx"] = df["diagnostic"]
    df["split"] = "external"                                      

    assert df["path"].notna().all(), "some images were not found"  
    return df[["image_id", "lesion_id", "dx", "label", "split", "path"]]