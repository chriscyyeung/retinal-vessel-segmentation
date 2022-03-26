import sys
import json
import argparse
import tensorflow as tf
from model import Model


def get_parser():
    parser = argparse.ArgumentParser(
        description="Implementation of the CENet architecture used for retinal "
                    "vessel segmentation. See https://arxiv.org/abs/1903.02740 "
                    "for more details."
    )
    parser.add_argument(
        "-p",
        "--phase",
        dest="phase",
        type=str,
        choices=["train", "test"],
        help="Training phase"
    )
    parser.add_argument(
        "--config_json",
        dest="config_json",
        type=str,
        default="config/config.json",
        help="JSON file for model configuration"
    )
    return parser.parse_args()


def main(FLAGS):
    # Load config file
    with open(FLAGS.config_json, "r") as config_json:
        config = json.load(config_json)

    # Set random seeds
    seed = config["Seed"]
    tf.random.set_seed(seed)

    # Run model
    model = Model(config)
    if FLAGS.phase == "train":
        model.train()
    elif FLAGS.phase == "test":
        model.test()
    else:
        sys.exit("Invalid training phase.")


if __name__ == '__main__':
    FLAGS = get_parser()
    main(FLAGS)
