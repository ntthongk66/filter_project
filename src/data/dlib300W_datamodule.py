from typing import Any, Dict, Optional, Tuple

import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from albumentations import Compose
import math
import matplotlib.pyplot as plt
from math import *
import torch
from torch.utils.data import Dataset

from src.data.components.dlib300W_Custom_dataset import DlibDataset, TransformDataset

class DLIB300WDataModule(LightningDataModule):
    def __init__(
        self,
        data_train: DlibDataset,
        data_test: DlibDataset,
        data_dir: str = "data\ibug_300W_large_face_landmark_dataset",
        train_val_test_split: Tuple[int, int] = (5666, 1000),
        transform_train: Optional[Compose] = None,
        transform_val : Optional[Compose] = None,
        batch_size = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        
        super().__init__()
        self.save_hyperparameters(logger = False)
            
            
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def prepare_data(self):
        pass
        
        
    def setup(self, stage: Optional[str] = None):
        
        # dataset = CustomDlibData(self.data_path, self.root_dir)
        
        if not self.data_train and not self.data_val and not self.data_test:
            
            dataset= self.hparams.data_train
                    # data_dir = self.hparams.data_d
            data_test = self.hparams.data_test
                    # data_dir = self.hparams.data_dir
                
            data_train, data_val = random_split(
                    dataset = dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(42)
                )
            
            self.data_train = TransformDataset(data_train, self.hparams.transform_train)
            self.data_val = TransformDataset(data_val, self.hparams.transform_val)
            self.data_test = TransformDataset(data_test, self.hparams.transform_val)
        
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm
    
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root"
    )
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    
    def test_dataset(cfg: DictConfig):
        dataset: DlibDataset = hydra.utils.instantiate(cfg.data_train)
        # dataset = data()
        print("dataset", len(dataset))
        image, landmarks = dataset[100]
        print("image", image.size, "landmarks", landmarks.shape)
        annotated_image = DlibDataset.annotate_image(image, landmarks)
        annotated_image.save(output_path / "test_dataset_result.png")
    
    def test_datamodule(cfg: DictConfig):
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        bx, by = next(iter(loader))
        
        print("n_batch", len(loader), bx.shape, by.shape, type(by))
        annotated_batch = TransformDataset.annotate_tensor(bx, by)
        print("annotated_batch", annotated_batch.shape)
        torchvision.utils.save_image(annotated_batch, output_path / "test_datamodule_result.png")
        
        for bx, by in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")
        
        for bx, by in tqdm(datamodule.val_dataloader()):
            pass
        print("training data passed")
        
        for bx, by in tqdm(datamodule.test_dataloader()):
            pass
        print("training data passed")

    @hydra.main(version_base = "1.3", config_path=config_path, config_name="dlib300w.yaml")
    def main(cfg: DictConfig):
        test_dataset(cfg)
        test_datamodule(cfg)
    
    main()    
    # _ = DLIB300WDataModule()

