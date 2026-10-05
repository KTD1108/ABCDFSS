from torchvision import transforms
from torch.utils.data import DataLoader

from .fss import FSS1000Dataset
from .isic import ISICDataset
from .lung import LungDataset
from .deepglobe import DeepglobeDataset
from .suim import SUIMDataset

DATASET_REGISTRY = {
    'fss': FSS1000Dataset,
    'fss1000': FSS1000Dataset,
    'isic': ISICDataset,
    'lung': LungDataset,
    'chest': LungDataset,
    'deepglobe': DeepglobeDataset,
    'suim': SUIMDataset
}

def build_dataloader(
    benchmark: str,
    datapath: str,
    shot: int = 1,
    img_size: int = 400,
    bsz: int = 1,
    nworker: int = 0,
    split: str = 'test',
    manifest_path: str = None
) -> DataLoader:
    """
    Unified dataset builder returning a standard PyTorch DataLoader.
    Supports optional pre-generated episode manifests for exact cross-experiment fairness.
    """
    key = benchmark.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown benchmark: '{benchmark}'. Available: {list(DATASET_REGISTRY.keys())}")

    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_cls = DATASET_REGISTRY[key]
    try:
        dataset = dataset_cls(datapath=datapath, transform=transform, shot=shot, split=split, manifest_path=manifest_path)
    except TypeError:
        dataset = dataset_cls(datapath=datapath, transform=transform, shot=shot, split=split)

    if len(dataset) == 0:
        raise RuntimeError(
            f"\n[!] Dataset returned 0 samples for benchmark '{benchmark}' at datapath '{datapath}'!\n"
            f"Please verify that the directory contains images or nested files."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=False,
        num_workers=nworker,
        pin_memory=True
    )
    return dataloader
