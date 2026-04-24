from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(data_dir="chest_xray", batch_size=16):

    # Absolute path handling — datasets live under <project_root>/datasets/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(data_dir):
        data_path = data_dir
    else:
        data_path = os.path.join(base_dir, "datasets", data_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Load dataset
    train_path = os.path.join(data_path, "train")

    full_dataset = datasets.ImageFolder(
        root=train_path,
        transform=transform
    )

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
