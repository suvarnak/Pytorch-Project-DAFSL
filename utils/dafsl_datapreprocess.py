import utils.dirs


"""
-Preprocess the domin dataset 
"""

import argparse
from utils.config import *


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)
    if config.data_mode == "imgs":
        img_root_folder = config.data_folder
        source_domain_list = config.data_domains.split(',')
        for src_dataset in source_domain_list:
            logging.info(
                "Processing the dataset for domain {}.".format(src_dataset))


if __name__ == "__main__":
    main()
