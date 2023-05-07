# encoding: utf-8
import os
import json
import shutil
import numpy as np
import yaml
import pkg_resources
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
logger = logging.getLogger()


def save_as_json(path, name, data_to_save):
    check_and_create_dir(path)
    # data_to_save = serialize_for_json(data_to_save)
    file_path = os.path.join(path, name)
    # print("Saving as json to: ", file_path)
    with open(file_path, 'w') as outfile:
        json.dump(data_to_save, outfile, cls=NumpyEncoder)


def load_as_json(path, name):
    # file_path = os.path.join(path, name)
    file_path = path + "/" + name
    # print("Load as json file: ", file_path)
    with open(file_path, 'r') as infile:
        data_loaded = json.load(infile)
    return data_loaded


def save_as_yaml(path, name, data_to_save):
    check_and_create_dir(path)
    file_path = os.path.join(path, name)
    print("Saving as yaml to: ", file_path)
    with open(file_path, 'w') as outfile:
        config = yaml.dump(data_to_save, outfile)
        return config


def load_as_yaml(path, name):
    file_path = os.path.join(path, name)
    print("Load as yaml from: ", file_path)
    with open(file_path, 'r') as infile:
        data_loaded = yaml.load(infile)
    return data_loaded


def serialize_for_json(data_to_serialize):
    if isinstance(data_to_serialize, set):
        print("Data saved is being converted from set() to list() data structure for JSON serialiszation !")
        return list(data_to_serialize)
    elif isinstance(data_to_serialize, np.ndarray):
        print("Data saved is being converted from np.ndarray() to list() data structure for JSON serialiszation !")
        return data_to_serialize.tolist()
    else:
        return data_to_serialize


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def copy_past_dir(src, dest):
    print("Updating files from ", src, " to ", dest)
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)


def get_extension(file_path):
    return file_path.split(".")[-1]


def list_dir(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return onlyfiles


def check_and_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Path did not exist. Creating path ", path)
        return False
    return True


def check_dir(path):
    if not os.path.isdir(path):
        return False
    return True


def load_dataset(dataset_name):
    # npz_file = pkg_resources.resource_filename("expert_finding", 'resources/{0}.npz'.format(dataset_name))
    npz_file = "/ddisk/lj/tmp/pycharm_project_16/impact_query_expert_finding/resources/dblp.npz"
    data = np.load(npz_file, allow_pickle=True)
    data_dict = dict()
    for k in data:
        if len(data[k].shape) == 0:
            data_dict[k] = data[k].flat[0]
        else:
            data_dict[k] = data[k]
        # logger.debug(
        #     f"{k:>10} shape = {str(data_dict[k].shape):<20}  "
        #     f"type = {str(type(data_dict[k])):<50}  "
        #     f"dtype = {data_dict[k].dtype}")

    A_da = data_dict["A_da"]
    A_dd = data_dict["A_dd"]
    T = data_dict["T"]
    L_d = data_dict["L_d"]
    L_d_mask = data_dict["L_d_mask"]
    L_a = data_dict["L_a"]
    L_a_mask = data_dict["L_a_mask"]
    tags = data_dict["tags"]

    return A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags
