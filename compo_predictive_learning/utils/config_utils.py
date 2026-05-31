import os 
import yaml
from omegaconf import OmegaConf
import collections
import numpy as np

def flatten(dictionary, parent_key=False, separator='.'):

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + str(key) if parent_key else str(key)
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        # elif isinstance(value, list):
        #     for k, v in enumerate(value):
        #         items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def get_folder_with_same_args(top_folder,other_args=None):
    same_folders = []
    for root, dirs, files in os.walk(top_folder):
        if '.hydra' in dirs:
            with open(os.path.join(root, '.hydra/config.yaml'), 'r') as f:
                saved_config = OmegaConf.load(f)
                saved_args = flatten(saved_config)
                # Check if saved_args has all key-value pairs in other_args
                if all(saved_args.get(k) == v for k, v in other_args.items()):
                    same_folders.append(root)
    return same_folders

def get_folder_of_same_params(top_folder,model_name,config,params_to_check):
    conf_to_check = {}
    flat_config = flatten(config)
    for key in params_to_check:
        conf_to_check[key] = flat_config[key]
    
    same_arg_folders = get_folder_with_same_args(top_folder,conf_to_check)
    for folder in same_arg_folders:
        if model_name in os.listdir(folder):
            return folder
    return ""

def get_results_for_hyperparams(params,parents_dir):

    results = []
    total_possible_results = 0
    for root, _, files in os.walk(parents_dir):
        if os.path.isfile(os.path.join(root,".hydra/config.yaml")) and  "results.npz" in files:
            saved_args = OmegaConf.load(os.path.join(root,".hydra/config.yaml"))
            same_params = True
            save_args_flat = flatten(saved_args)
            total_possible_results += 1
            for k, v in params.items():
                if k not in save_args_flat:
                    if isinstance(v,bool) and v == False:
                        continue
                    elif isinstance(v,bool) and v == True:
                        same_params = False
                        break
                    elif isinstance(v,float) and v == 0.0:
                        continue
                    elif isinstance(v,float) and v == 0.0:
                        same_params = False
                        break
                    else:
                        same_params = False
                        break
                if save_args_flat[k] != v:
                    same_params = False
                    break
            if same_params:
                results.append( (save_args_flat,root,np.load(os.path.join(root,"results.npz"),allow_pickle=True)) )
    print("Total possible results: ",total_possible_results)
    print("Total results found: ",len(results))
    return results

def average_seed(results):
    grouped_results = collections.defaultdict(list)
    for params, path,data in results:
        # Remove 'seed' from parameters to create a grouping key
        params_without_seed = frozenset((k, v) for k, v in params.items() if k != "seed")
        grouped_results[params_without_seed].append((params,data,path))
            
    grouped_concatenated_results = []
    for  gres in grouped_results.values():
        data = {}
        for res in gres:
            for key in res[1].files:
                if key not in data:
                    data[key] = []
                data[key].append(res[1][key])
        data = {key: np.stack(val,axis=0) for key,val in data.items()}
        path = gres[0][2]
        hparams = gres[0][0] 
        grouped_concatenated_results.append((hparams,path,data))
    return grouped_concatenated_results

def filter_results(results,filter_params):
    return [res for res in results if all(res[0].get(k) in v for k, v in filter_params.items())]

def group_results_by_params(results,param_names):
    if isinstance(results,dict):
        new_dict = {}
        for k in results:
            new_dict[k] = group_results_by_params(results[k],param_names)
        return new_dict
    if isinstance(results,list):
        if len(param_names) > 1:
            results_dict = {}
            for res in results:
                if res[0][param_names[0]] not in results_dict:
                    results_dict[res[0][param_names[0]]] = []
                results_dict[res[0][param_names[0]]].append(res)
            for k in results_dict:
                results_dict[k] = group_results_by_params(results_dict[k],param_names[1:])
            return results_dict
        results_dict = {}
        for res in results:
            if param_names[0] not in res[0]:
                print("Error in ",res[1])
                continue
            if res[0][param_names[0]] not in results_dict:
                results_dict[res[0][param_names[0]]] = []
            results_dict[res[0][param_names[0]]].append(res)
            print(res[0][param_names[0]])
        return results_dict

def all_common_keys(list_of_dicts):
    common_keys = set(list_of_dicts[0].keys())
    for d in list_of_dicts:
        common_keys = common_keys.intersection(set(d.keys()))
    return common_keys

def get_data_out_from_npz_and_flatten(results):
    if isinstance(results,dict):
        new_dict = {}
        for k in results:
            new_dict[k] = get_data_out_from_npz_and_flatten(results[k])
        return new_dict
    if isinstance(results,list):
        
        flat_dict = {"hyperparameters":[],"paths":[]}
        common_curves = all_common_keys([r[2] for r in results])
        for res_name in common_curves:
            flat_dict[res_name] = []
        for res in results:
            for res_name in common_curves:
                flat_dict[res_name].append(res[2][res_name])
            flat_dict["hyperparameters"].append(res[0])
            flat_dict["paths"].append(res[1])

        for res_name in common_curves:
            flat_dict[res_name] = np.array(flat_dict[res_name])
        return flat_dict

        