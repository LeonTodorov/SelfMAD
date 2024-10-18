import torch
import numpy as np
from tqdm import tqdm
from utils.model import Detector
import numpy as np
from utils.dataset import PartialMorphDataset
# from utils.dataset import MorDIFF
from utils.metrics import calculate_eer, calculate_auc
import json
import argparse

def main(eval_config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_state_path = eval_config["model_path"]
    model = Detector(model=eval_config["model_type"])
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state['model'])
    model.train(mode=False)
    model.to(device)

    image_size = 384 if "hrnet" in eval_config["model_type"] else 380
    batch_size = 32

    # ORIGINAL DATASETS (FRGC, FERET, FRLL)
    test_datasets = default_datasets(image_size, datasets="original", config=eval_config)
    test_loaders = prep_dataloaders(test_datasets, batch_size)

    if eval_config["verbose"]:
        for dataset in test_datasets:
            print(f'-----{dataset}:')
            for method in test_datasets[dataset]:
                print('-', method)
                print("real", test_datasets[dataset][method].labels.count(0))
                print("fake", test_datasets[dataset][method].labels.count(1))
                
    evaluate(model, test_loaders, device, calculate_means=True, verbose=True)

    # MORDIFF DATASET
    # test_datasets = default_datasets(image_size, datasets="MorDIFF", config=eval_config)
    # test_loaders = prep_dataloaders(test_datasets, batch_size)
    
    # if eval_config["verbose"]:
    #     for dataset in test_datasets:
    #         print(f'-----{dataset}:')
    #         for method in test_datasets[dataset]:
    #             print('-', method)
    #             print("real", test_datasets[dataset][method].labels.count(0))
    #             print("fake", test_datasets[dataset][method].labels.count(1))
                
    # evaluate(model, test_loaders, device, calculate_means=False, verbose=True)
    
    
def default_datasets(image_size, datasets="original", config=None):
    assert datasets in ["original", 
                        # "MorDIFF"
                        ]
    if datasets == "original":
        FRGC_datapath = config["FRGC_path"]
        FERET_datapath = config["FERET_path"]
        FRLL_datapath = config["FRLL_path"]

        test_datasets = {
            "FRGC": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='stylegan')
            },
            "FERET": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='stylegan')
            },
            "FRLL": {
                "amsl": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='amsl'),
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='stylegan'),
                "wm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='webmorph')
            }
        }
        return test_datasets
    # if datasets == "MorDIFF":
    #     test_datasets = {
    #         "MorDIFF": {
    #             "MorDIFF": MorDIFF(datapath_fake=config["MorDIFF_f_path"],
    #                                 datapath_real=config["MorDIFF_bf_path"],
    #                                 image_size=image_size)
    #         }
    #     }
    #     return test_datasets
    
def prep_dataloaders(test_datasets, batch_size):
    test_loaders = {
        dataset: {
            method: torch.utils.data.DataLoader(test_datasets[dataset][method],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
            for method in test_datasets[dataset]
        }
        for dataset in test_datasets
    }
    return test_loaders

def evaluate(model, test_loaders, device, calculate_means=True, verbose=False, multi=False):
    results = {}
    if calculate_means:
        total_eers, total_aucs = [], []
    for dataset_loader in test_loaders:
        for method_loader in test_loaders[dataset_loader]:
            output_dict = []
            target_dict = []
            for data in tqdm(test_loaders[dataset_loader][method_loader], desc=f"Evaluating {dataset_loader}_{method_loader}"):
                img = data[0].to(device, non_blocking=True).float()
                target = data[1].to(device, non_blocking=True).long()
                with torch.no_grad():
                    output = model(img)
                if multi:
                    output = torch.cat((output[:, 0].unsqueeze(1), output[:, 1:].sum(1).unsqueeze(1)), dim=1)
                output_dict += output.softmax(1)[:, 1].cpu().data.numpy().tolist()
                target_dict += target.cpu().data.numpy().tolist()
            eer = calculate_eer(target_dict, output_dict)
            auc = calculate_auc(target_dict, output_dict)
            if calculate_means:
                total_eers.append(eer)
                total_aucs.append(auc)
            if verbose:
                print(f"{dataset_loader}_{method_loader} auc: {auc:.4f}, eer: {eer:.4f}")
            results[f"{dataset_loader}_{method_loader}"] = {"auc": auc, "eer": eer}
    if calculate_means:
        if verbose:
            print(f"Total mean auc: {np.mean(total_aucs):.4f}, mean eer: {np.mean(total_eers):.4f}")
        results["mean"] = {"auc": np.mean(total_aucs), "eer": np.mean(total_eers)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Print more information')
    parser.add_argument('-m', dest='model_type', type=str, required=True, help='Type of the model, e.g. hrnet_w18')
    parser.add_argument('-p', dest='model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('-FRLL_path', type=str, required=False)
    parser.add_argument('-FRGC_path', type=str, required=False)
    parser.add_argument('-FERET_path', type=str, required=False)
    # parser.add_argument('-MorDIFF_f_path', type=str, required=False)
    # parser.add_argument('-MorDIFF_bf_path', type=str, required=False)
    args = parser.parse_args()

    eval_config = json.load(open("./configs/data_config.json"))
    for key in vars(args):
        if vars(args)[key] is not None:
            eval_config[key] = vars(args)[key]

    main(eval_config)
