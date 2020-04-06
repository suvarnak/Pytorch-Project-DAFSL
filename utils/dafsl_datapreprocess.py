import logging
import os
import utils.dirs
import json
import shutil
import glob
"""
-Preprocess the domin dataset
"""

import argparse
from utils.config import *
import pandas as pd

'''
from the downloaded aircraft dataset, cretes the train test valid dir structure  as per splits in meta-dataset
'''


def create_dir(dirname):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            return True
        else:
            logging.info("directory exists"+dirname)
            return False
    except Exception as err:
        logging.getLogger("Dirs Creator").info(
            "Creating directories error: {0}".format(err))
        exit(-1)
    return False



def get_imgs_for_class_aircraft(dataset_root_dir, train_class_dir, split="train"):
    if split != "valid":
		    trainimgs_mapping_filename = os.path.join(dataset_root_dir, "data/images_variant_"+split+".txt")
    else:
	    trainimgs_mapping_filename = os.path.join(
		        dataset_root_dir, "data/images_variant_"+"val"+".txt")

    train_imgs_df = pd.read_csv(
        trainimgs_mapping_filename, sep=" ", names=['image', 'label'], dtype=str)
    train_imgs_df = train_imgs_df.set_index(['label'])
    trainclass_imgs_df = train_imgs_df.loc[train_class_dir]
    image_list = trainclass_imgs_df['image'].tolist()
    # print(image_list)
    return image_list


def get_imgs_for_class_cu_birds(dataset_root_dir, train_class_dir, split="train"):
    print("!!!!!!!!!",train_class_dir)
    included_extensions = ['jpg','jpeg']
    img_path = os.path.join(dataset_root_dir, "images", train_class_dir)
    image_list = glob.glob(img_path+'/[!._]*.jpg')
    img_names_list = []
    for fn in image_list:
			  img_names_list.append(os.path.basename(fn))
    print(img_names_list)
    return img_names_list


def get_imgs_for_class_omniglot(dataset_root_dir, train_class_dir, split="train"):
    image_list = os.listdir(os.path.join(
        dataset_root_dir, "images/Sanskrit", train_class_dir))
    # print(image_list)
    return image_list


def copy_files_omniglot(src, dst, imglist):
    print("SRC", src)
    print("DST", dst)
    for fname in imglist:
        filename = ""
        filename = str(fname) 
        src1 = os.path.join(src, filename)
        dst1 = os.path.join(dst, filename)
        shutil.copyfile(src1, dst1)


def copy_files(src, dst, imglist,extension=".jpg"):
    print("SRC", src)
    print("DST", dst)
    for fname in imglist:
        filename = ""
        filename = str(fname) + extension
        src1 = os.path.join(src, filename)
        dst1 = os.path.join(dst, filename)
        shutil.copyfile(src1, dst1)


def preprocess_data(config, src_domain_name, spec="train"):
    logging.info(
        "Processing the dataset for domain {}.".format(src_domain_name))
    src_dataset_spec = os.path.join(
        config.data_spec_folder, src_domain_name + "_splits.json")
    with open(src_dataset_spec) as f:
        data = json.load(f)
    dataset_root_dir = os.path.join(config.datasets_root_dir, src_domain_name)
    print(dataset_root_dir)
    dataset_img_dir = os.path.join(
        config.datasets_root_dir, src_domain_name,  config.domains_img_dir[src_domain_name])
    already_processed = not create_dir(os.path.join(dataset_root_dir, spec))
    if already_processed:
        return
    dir_list = data[spec]
    logging.info(
        "training classes list for domain {}.".format(dir_list))
    for class_dir in dir_list:
		    create_dir(os.path.join(dataset_root_dir, spec, class_dir))
    for class_dir in dir_list:
        if src_domain_name == "aircraft":
            train_imglist = get_imgs_for_class_aircraft(dataset_root_dir, class_dir)
            copy_files(dataset_img_dir, os.path.join(dataset_root_dir, spec, class_dir), train_imglist,extension=".jpg")
        elif src_domain_name == "CUB_200_2011":
            train_imglist = get_imgs_for_class_cu_birds(dataset_root_dir, class_dir)
            src_dir = os.path.join(dataset_root_dir,"images",class_dir)
            print(src_dir)
            copy_files(src_dir, os.path.join(dataset_root_dir, spec, class_dir), train_imglist,extension="")
        elif src_domain_name == "omniglot":
            train_imglist = get_imgs_for_class_omniglot(dataset_root_dir, class_dir)
            src_dir = os.path.join(dataset_root_dir,"images/Sanskrit",class_dir)
            print(src_dir)
            copy_files_omniglot(src_dir, os.path.join(dataset_root_dir, spec, class_dir), train_imglist)


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
        source_domain_list = config.data_domains.split(',')
        for src_domain_name in source_domain_list:
            preprocess_data(config, src_domain_name, spec="train")
            preprocess_data(config, src_domain_name, spec="test")
            preprocess_data(config, src_domain_name, spec="valid")


if __name__ == "__main__":
    main()
