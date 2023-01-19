import wandb
from omegaconf import OmegaConf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--run_path", type=str, default="dtumlops-group19/backbones/3nm3vv75"
)
parser.add_argument("--config_path", type=str, default="config/wandb_config.yaml")
args = parser.parse_args()

api = wandb.Api()
run = api.run(args.run_path)
config = OmegaConf.create(run.config)
OmegaConf.save(config=config, f=args.config_path)
