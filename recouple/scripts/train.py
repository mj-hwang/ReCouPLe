import argparse
import os
import subprocess
import datetime
import random

from research.utils.config import Config


def try_wandb_setup(path, config):
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None and wandb_api_key != "":
        try:
            import wandb
        except ImportError:
            return
        project_dir = os.path.dirname(os.path.dirname(__file__)) + "_maniskill_sac_final"
        print("project:", os.path.basename(project_dir))
        print("name:", os.path.basename(path))
        print("config:", config.flatten(separator="-"))
        print("dir:", os.path.join(os.path.dirname(project_dir), "wandb"))

        wandb.init(
            project=os.path.basename(project_dir),
            name=os.path.basename(path),
            entity='rpl-experiments',
            config=config.flatten(separator="-"),
            dir=os.path.join(os.path.dirname(project_dir), "wandb"),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = Config.load(args.config)
    config.config["seed"] = args.seed

    relabel_data_paths = config.get("relabel_data_paths", None)
    new_data_paths = config.get("new_data_paths", None)
    
    savepath = os.path.join(args.path, f'{config["alg"]}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    savepath = savepath + f"_{random.randint(1, 10000)}" # to prevent overwriting the same experiment for sweeps
    print(f"Saving to {savepath}")
    os.makedirs(args.path, exist_ok=True)
    os.makedirs(savepath, exist_ok=False)  # Change this to false temporarily so we don't recreate experiments
    try_wandb_setup(savepath, config)
    config.save(savepath)  # Save the config
    # # save the git hash
    # process = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
    # git_head_hash = process.communicate()[0].strip()
    # with open(os.path.join(args.path, "git_hash.txt"), "wb") as f:
    #     f.write(git_head_hash)
    # Parse the config file to resolve names.
    config = config.parse()
    # Get everything at once.
    trainer = config.get_trainer(device=args.device)
    # Train the model
    trainer.train(savepath)

    if relabel_data_paths is not None:
        for relabel_data_path, new_data_path in zip(relabel_data_paths, new_data_paths):
            print(f"Relabeling data at {relabel_data_path} with new data at {new_data_path}")
            trainer.label_reward(relabel_data_path, new_data_path)
