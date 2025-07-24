import os
from glob import glob
import yaml
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active_checkpoint",
        type=str,
        default=None,
        help="path to the active checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="if using active_checkpoint, this is ignored because the config is read from the checkpoint directory",
    )
    parser.add_argument(
        "--save_results_at",
        type=str,
        required=True,
        help="csv file to save numerical results as a row with model name and result entries. Other kinds of results will be saved in the checkpoint directory",
    )

    parser.add_argument(
        "--future_pred_acc",
        action="store_true",
        help="whether to compute future prediction accuracy",
    )
    parser.add_argument(
        "--recon_acc",
        action="store_true",
        help="whether to compute future prediction accuracy",
    )
    parser.add_argument(
        "--recon_waveform",
        action="store_true",
    )
    parser.add_argument(
        "--future_pred_waveform",
        action="store_true",
        help="whether to compute future prediction accuracy on waveform level",
    )

    parser.add_argument("--debug", action="store_true")
    # ........... can add more arguments here ...........

    # Parse known and unknown arguments
    known_args, unknown_args = parser.parse_known_args()

    # find the config file based on the active_checkpoint
    config_path = glob(
        os.path.join(
            os.path.dirname(known_args.active_checkpoint)
            if known_args.active_checkpoint
            else ".",
            "*.yaml",
        )
    )
    print(f"Config files found: {config_path}")
    known_args.config = (
        config_path[0] if config_path else known_args.config
    )  # use the first found config file if active_checkpoint is provided

    # Process unknown arguments as key-value pairs
    additional_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            value = unknown_args[unknown_args.index(arg) + 1]
            additional_args[key] = value

    with open(known_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["active_checkpoint"] = known_args.active_checkpoint

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

    if known_args.debug > 0:
        config["debug"] = True

    # ready for testing
    if "ISymm" in config["method"]:
        from tester import Tester

    tester = Tester(config)
    tester.prepare_data()
    tester.build_model()
    tester.test(
        future_pred_acc=known_args.future_pred_acc,
        recon_waveform=known_args.recon_waveform,
        future_pred_waveform=known_args.future_pred_waveform,
        vis_tsne=config.get("vis_tsne", False),
        confusion_mtx=config.get("confusion_mtx", False),
    )
