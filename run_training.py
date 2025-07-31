import os
import yaml
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    # ........... can add more arguments here ...........

    # Parse known and unknown arguments
    known_args, unknown_args = parser.parse_known_args()

    # Process unknown arguments as key-value pairs
    additional_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            value = unknown_args[unknown_args.index(arg) + 1]
            additional_args[key] = value

    with open(known_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update config with additional arguments
    for key, value in additional_args.items():
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        if "." in key:
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
            print(f"Set {key} to {value}")
            # enable arbitrary nested keys

        else:
            config[key] = value
            print(f"Set {key} to {value}")

    if known_args.name is not None:
        config["name"] = known_args.name

    if known_args.debug > 0:
        config["debug"] = True

    if config["method"] == "ISymm":
        from trainer import Trainer
    elif config["method"] == "ISymm_Induced":
        from trainer_induced import TrainerInduced as Trainer

    trainer = Trainer(config)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
