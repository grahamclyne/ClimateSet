from emulator.src.core.models.basemodel import BaseModel
from tensordict.tensordict import TensorDict
from omegaconf import DictConfig
import torch
from hydra.utils import instantiate
# from DecadeCast.model.forecast import ForecastModuleWithCond
from geoarches.backbones.dit import TimestepEmbedder

class ArchesClimate(BaseModel):
    def __init__(
        self,
        datamodule_config: DictConfig = None,
        model_config: DictConfig = None,
        pretrained_path: str = None,

        *args,
        **kwargs,
    ):
        # print('model_config',model_config)
        super().__init__(datamodule_config=datamodule_config,model_config=model_config,**kwargs)
        self.save_hyperparameters()
        self.__dict__.update(locals())
        # self.model = instantiate(model_config.module.forecast,model_config.module)
        self.backbone = instantiate(model_config.module.backbone)  # necessary to put it on device
        self.embedder = instantiate(model_config.module.embedder)
        # self.learn_residual='default'
        # self.state_normalization=False
        # self.prediction_type='sample'
        if(pretrained_path is not None):
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
            self.load_state_dict(checkpoint["state_dict"], strict=False)

    def forward(self, X):
        x = self.embedder.encode(X)
        embedder = TimestepEmbedder(512)
        cond_emb = embedder(torch.tensor([1])).to(X.device) # dummy value for now 
        x = self.backbone(x,cond_emb)
        out = self.embedder.decode(x)  
        return out

