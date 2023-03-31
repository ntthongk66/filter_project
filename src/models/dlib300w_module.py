from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.regression.mae import MeanAbsoluteError


class DlibLiModule(LightningModule):
    def __init__ (
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])
        
        self.net = net
        
        #loss func
        self.criterion = torch.nn.MSELoss()
        
        #metric objects for calculating and averaging accuracy across batches
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
                
        #metric
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        #tracking best so far validation acc
        self.val_mae_best = MinMetric()
        
    def forward(self, x:torch.Tensor):
        return self.net(x)
    
    def on_train_start(self):
        self.val_mae_best.reset()
    
    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y
    
    
    def training_step(self, batch:Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        
        #update log and metrics
        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train/loss", self.train_loss, on_step= False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step= False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def training_epoch_end(self, outputs: List[Any]):
        pass
        
    
    def validation_step(self, batch:Any, batch_idx:int):
        loss, preds, targets = self.model_step(batch)
        
        # update and log metrics
        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
        
    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_mae.compute() #get current val acc
        self.val_mae_best(acc)
        
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True, sync_dist=True)
        
        
        
        
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
    def predict_step(self, batch:Any, batch_idx:int, dataloader_idx: int = 0) -> Any:
        _, preds, _ = self.model_step(batch)
        return preds

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
if __name__ == "__main__":
   # read config file from configs/model/dlib_resnet.yaml
    import pyrootutils
    from omegaconf import DictConfig
    import hydra

    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    output_path = path / "outputs"
    print("paths", path, config_path, output_path)

    def test_net(cfg):
        net = hydra.utils.instantiate(cfg.net)
        print("*"*20+" net "+"*"*20, "\n", net)
        output = net(torch.randn(16, 3, 224, 224))
        print("output", output.shape)

    def test_module(cfg):
        module = hydra.utils.instantiate(cfg)
        output = module(torch.randn(16, 3, 224, 224))
        print("module output", output.shape)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="dlib_resnet.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_net(cfg)
        test_module(cfg)

    main()