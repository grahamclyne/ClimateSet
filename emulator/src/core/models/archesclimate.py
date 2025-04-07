from emulator.src.core.models.basemodel import BaseModel
from tensordict.tensordict import TensorDict
from omegaconf import DictConfig
import torch
from hydra.utils import instantiate
from DecadeCast.model.forecast import ForecastModuleWithCond

class ArchesClimate(BaseModel):
    def __init__(
        self,
        datamodule_config: DictConfig = None,
        model_config: DictConfig = None,
        forecast=None,
        *args,
        **kwargs,
    ):
        print('kwargs',kwargs)
        print('datamodule_config',datamodule_config)
        print('model_config',model_config)
        super().__init__(datamodule_config=datamodule_config,**kwargs)
        self.save_hyperparameters()
        self.__dict__.update(locals())
        forecast_model = instantiate(forecast,cfg=model_config)
        # self.learn_residual='default'
        # self.state_normalization=False
        # self.prediction_type='sample'

        
    #training step needs to be a frankensteinization of both from ClimateSet and ArchesClimate
    def training_step(self, batch, batch_nb: int):


        print(batch[0].shape)
        print(len(batch))
        if self.super_emulation:
            X, Y, idx = batch
        else:
            X, Y = batch
            idx = None
        pred = forecast_model.forward(X)




        
        # preds = self.predict(X, idx)
        # dict with keys being the output var ids
        Y = self.output_postprocesser.split_vector_by_variable(
            Y
        )  # split per var id #TODO: might need to remove that for other datamodule

        train_log = dict()  # everything we want to log to wandb should go in here

        loss = 0

        #  Loop over output variable to compute loss seperateley!!!
        for out_var in self._out_var_ids:
            loss_per_var = self.criterion(preds[out_var], Y[out_var])
            if torch.isnan(loss_per_var).sum() > 0:
                exit(0)
            loss += loss_per_var
            train_log[f"train/{out_var}/loss"] = loss_per_var
            # any additional losses can be computed, logged and added to the loss here

        # Average Loss over vars
        loss = loss / len(self._out_var_ids)

        n_zero_gradients = (
            sum(
                [
                    int(torch.count_nonzero(p.grad == 0))
                    for p in self.parameters()
                    if p.grad is not None
                ]
            )
            / self.n_params
        )

        self.log_dict(
            {**train_log, "train/loss": loss, "n_zero_gradients": n_zero_gradients}
        )

        ret = {
            "loss": loss,
            "n_zero_gradients": n_zero_gradients,
            "targets": Y,
            "preds": preds,
        }

        return ret










         # def training_step(self, batch, batch_nb):
        # sample timesteps
        device, bs = batch["state"].device, batch["state"].shape[0]

        loss = self.loss(pred, target_state, timesteps)

        self.mylog(loss=loss)

        return loss

