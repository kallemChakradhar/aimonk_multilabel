import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MultiLabelDataset(Dataset):
    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir

        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        raw_samples = []

        for line in lines:
            parts = line.split()
            img_name = parts[0]
            labels = parts[1:]

            label_tensor = []
            mask_tensor = []

            for val in labels:
                if val == "NA":
                    label_tensor.append(0.0)
                    mask_tensor.append(0.0)  # ignore in loss
                else:
                    label_tensor.append(float(val))
                    mask_tensor.append(1.0)

            raw_samples.append(
                (img_name,
                 torch.tensor(label_tensor),
                 torch.tensor(mask_tensor))
            )

        #  FILTER MISSING FILES (Professional approach)
        self.samples = []
        for img_name, labels, mask in raw_samples:
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                self.samples.append((img_name, labels, mask))
            else:
                print(f"Missing file skipped: {img_name}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, labels, mask = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Skip corrupted image
            return self.__getitem__((idx + 1) % len(self.samples))

        image = self.transform(image)

        return image, labels, mask
