from omegaconf import OmegaConf,DictConfig
import hydra
from emulator.src.utils.interface import get_model,get_datamodule
import torch
@hydra.main(config_path="emulator/configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig,overrides=['datamodule.batch_size=1']):
    #load training data
    # X_test, y_test, (so2_solver, bc_solver) = load_test_data('test', None)
    # test_data = ClimateDataset(X_test, y_test)
    # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    #load model
    emulator_model = get_model(config)
    emulator_model.to('cuda')
    data_module = get_datamodule(config)
    data_module.setup('test')
    test = data_module.test_dataloader()[0]
    output = []
    real_output = []
    for X,y in test:
        with torch.no_grad():
            out = emulator_model(X.to('cuda'))
            output.append(out)
            torch.cuda.empty_cache()
            target_output.append(y)
    torch.save(torch.stack(output),'test_output',)
    torch.save(torch.stack(target_output),'test_output_target',)

main()
    
