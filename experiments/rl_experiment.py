from utils.config import ExperimentConfig
from data_generators.data_generators import DataGenerator
from models.mlp_sales import MLPLogSalesModel
from env.gelateria_env import GelateriaEnv
from env.reward.simple_reward import SimpleReward


def get_experiment_config():
    config = ExperimentConfig()
    return config


class SupervisedExperiment:

    def __init__(self):
        self._config: ExperimentConfig = get_experiment_config()

    def _get_dataset_generator(self):
        return DataGenerator(config=self._config.data_generation_config,
                             dataloader_config=self._config.dataloader_config)

    def load_model(self):
        dummy_dataloader, _ = self._get_dataset_generator().get_train_val_dataloaders()
        input_dim = next(iter(dummy_dataloader))["public_obs"].shape[-1]
        model = MLPLogSalesModel(input_dim=input_dim, name="mlp_sales", config=self._config.net_config)
        model.load()
        return model

    def build_env(self):
        reward = SimpleReward(waste_penalty=0.0)
        env = GelateriaEnv(init_state=None,
                           sales_model=self.load_model(),
                           reward=reward,
                           restock_fct=None)
        return env

    def get_time_spec(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    SupervisedExperiment().run()
