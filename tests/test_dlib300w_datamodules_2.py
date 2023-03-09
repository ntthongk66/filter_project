import hydra 
import omegaconf
import pyrootutils
import matplotlib.pyplot as plt


root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "dlib300w.yaml")

dlib300w_datamodule = hydra.utils.instantiate(cfg)
dlib300w_datamodule.setup()

train_loader = dlib300w_datamodule.train_dataloader()
dlib300w_datamodule.draw_batch(next(iter(train_loader)), 32)