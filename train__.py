import os
import torch
import torch.nn as nn
import numpy as np
import os
import random
from utils.selfMAD import selfMAD_Dataset
from utils.scheduler import LinearDecayLR
import argparse
from utils.logs import log
from datetime import datetime
from tqdm import tqdm
from model import Detector
import json
from utils.metrics import calculate_eer, calculate_auc
from eval__ import default_datasets, prep_dataloaders, evaluate

def main(args):

    assert args["model"] in ["efficientnet-b4", "efficientnet-b7", "swin", "resnet", "hrnet_w18", "hrnet_w32", "hrnet_w44", "hrnet_w64"]
    assert args["train_dataset"] in ["FF", "SMDD"]
    assert args["saving_strategy"] in ["original", "testset_best"]

    # FOR REPRODUCIBILITY
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = {
        "session_name": args["session_name"],
        "train_dataset": args["train_dataset"],
        "model": args["model"],
        "epochs": args["epochs"],
        "batch_size": args["batch_size"],
        "learning_rate": args["lr"],
        "image_size": 384 if "hrnet" in args["model"] else 380,
        "saving_strategy": args["saving_strategy"],
    }

    device = torch.device('cuda')

    # if args.train_dataset == "FF":
    #     train_datapath = val_datapath = '/mnt/hdd/leon/FF++_10/old_FF++_every_10_frame'
    # elif args.train_dataset == "SMDD":
    #     train_datapath = "/mnt/hdd/leon/SMDD_release_train/os25k_bf_t/"
    
    # No validation set in this case.
    if args["train_dataset"] == "SMDD":
        assert args["saving_strategy"] == "testset_best"
    
    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    train_datapath = args["SMDD_path"] if args["train_dataset"] == "SMDD" else args["FF_path"]
    train_dataset=selfMAD_Dataset(phase='train',image_size=image_size, datapath=train_datapath)
    if args["train_dataset"] == "FF":
        val_dataset=selfMAD_Dataset(phase='val',image_size=image_size, datapath=train_datapath)

    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )
    if args["train_dataset"] == "FF":
        val_loader=torch.utils.data.DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=val_dataset.collate_fn,
                            num_workers=4,
                            pin_memory=True,
                            worker_init_fn=val_dataset.worker_init_fn
                            )
    
    test_datasets = default_datasets(image_size, datasets="original", config={
        "FRLL_path": args["FRLL_path"],
        "FRGC_path": args["FRGC_path"],
        "FERET_path": args["FERET_path"]
    })
    test_datasets_mordiff = default_datasets(image_size, datasets="MorDIFF", config={
        "MorDIFF_f_path": args["MorDIFF_f_path"],
        "MorDIFF_bf_path": args["MorDIFF_bf_path"]
    })
    test_loaders = prep_dataloaders(test_datasets, batch_size)
    test_loaders_mordiff = prep_dataloaders(test_datasets_mordiff, batch_size)
    

    model=Detector(model=args["model"], lr=args["lr"])
    model=model.to('cuda')
    n_epoch=cfg['epochs']
    lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    
    now=datetime.now()
    # local
    # save_path='/mnt/hdd/leon/models'
    save_path='{}/{}'.format(args["save_path"], args["session_name"])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    os.mkdir(save_path)
    os.mkdir(save_path+'weights/')
    os.mkdir(save_path+'logs/')
    with open(save_path+"config.txt", "w") as f:
        f.write(str(cfg))
    logger = log(path=save_path+"logs/", file="losses.logs")

    criterion=nn.CrossEntropyLoss()
    if args["saving_strategy"] == "original":
        last_val_auc=0
        weight_dict={}
        n_weight=5
    elif args["saving_strategy"] == "testset_best":
        best_mean = None
        best_epoch = None
    for epoch in range(n_epoch):
        # TRAIN LOOP ##################################################
        np.random.seed(seed + epoch)
        train_loss=0.
        model.train(mode=True)
        for data in tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, n_epoch)):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            output=model.training_step(img, target)
            loss=criterion(output,target)
            train_loss+=loss.item()
        lr_scheduler.step()
        
        log_text="Epoch {}/{} | train loss: {:.4f} |".format(
                        epoch+1,
                        n_epoch,
                        train_loss/len(train_loader),
                        )
        # VAL LOOP ##################################################
        if args["train_dataset"] == "FF":
            model.train(mode=False)
            output_dict=[]
            target_dict=[]
            np.random.seed(seed)
            for data in tqdm(val_loader, desc="Running validation"):
                img=data['img'].to(device, non_blocking=True).float()
                target=data['label'].to(device, non_blocking=True).long()
                with torch.no_grad():
                    output=model(img)
                output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
                target_dict+=target.cpu().data.numpy().tolist()
            val_auc=calculate_auc(target_dict,output_dict)
            val_eer=calculate_eer(target_dict,output_dict)
            log_text+=" val auc: {:.4f}, val eer: {:.4f} |".format(
                            val_auc,
                            val_eer
            )
        # TEST LOOP ###################################################
        model.train(mode=False)
        results_original_dataset = evaluate(model, test_loaders, device, calculate_means=True)
        results_mordiff_dataset = evaluate(model, test_loaders_mordiff, device, calculate_means=False)
        for dataset in results_original_dataset:
            log_text += f" {dataset}: auc: {results_original_dataset[dataset]['auc']:.4f}, eer: {results_original_dataset[dataset]['eer']:.4f} |"
        for dataset in results_mordiff_dataset:
            log_text += f" {dataset}: auc: {results_mordiff_dataset[dataset]['auc']:.4f}, eer: {results_mordiff_dataset[dataset]['eer']:.4f}"
        # SAVE MODEL ###################################################
        if args["saving_strategy"] == "original":
            if len(weight_dict)<n_weight:
                save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                weight_dict[save_model_path]=val_auc
                torch.save({
                        "model":model.state_dict(),
                        "optimizer":model.optimizer.state_dict(),
                        "epoch":epoch
                    },save_model_path)
                last_val_auc=min([weight_dict[k] for k in weight_dict])

            elif val_auc>=last_val_auc:
                save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
                for k in weight_dict:
                    if weight_dict[k]==last_val_auc:
                        del weight_dict[k]
                        os.remove(k)
                        weight_dict[save_model_path]=val_auc
                        break
                torch.save({
                        "model":model.state_dict(),
                        "optimizer":model.optimizer.state_dict(),
                        "epoch":epoch
                    },save_model_path)
                last_val_auc=min([weight_dict[k] for k in weight_dict])
        elif args["saving_strategy"] == "testset_best":
            if best_mean is None or results_original_dataset['mean']['eer'] < best_mean:
                best_mean = results_original_dataset['mean']['eer']
                # remove previous best model
                if os.path.exists(os.path.join(save_path, "epoch_{}.tar".format(best_epoch))):
                    os.remove(os.path.join(save_path, "epoch_{}.tar".format(best_epoch)))
                best_epoch = epoch + 1
                save_model_path=os.path.join(save_path+'weights/',"epoch_{}.tar".format(best_epoch))
                torch.save({
                        "model":model.state_dict(),
                        "optimizer":model.optimizer.state_dict(),
                        "epoch":epoch
                    },save_model_path)
        logger.info(log_text)
        
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # require name of run
    parser.add_argument('-n',dest='session_name', type=str, required=True)
    # specified in train_config.json, but can be overriden here
    parser.add_argument('-m',dest='model',type=str,required=False)
    parser.add_argument('-b', dest='batch_size', type=int, required=False)
    parser.add_argument('-e', dest='epochs', type=int, required=False)
    parser.add_argument('-v', dest='saving_strategy', type=str, required=False)
    parser.add_argument('-t', dest='train_dataset', type=str, required=False)
    # parser.add_argument('-p', dest='train_datapath', type=str, required=False)
    parser.add_argument('-s', dest='save_path', type=str, required=False)
    parser.add_argument('-lr', dest='lr', type=float, required=False)
    parser.add_argument('-FRLL_path', type=str, required=False)
    parser.add_argument('-FRGC_path', type=str, required=False)
    parser.add_argument('-FERET_path', type=str, required=False)
    parser.add_argument('-MorDIFF_f_path', type=str, required=False)
    parser.add_argument('-MorDIFF_bf_path', type=str, required=False)
    parser.add_argument('-SMDD_path', type=str, required=False)
    parser.add_argument('-FF_path', type=str, required=False)

    args=parser.parse_args()

    train_config = json.load(open("train_config.json"))
    for key in vars(args):
        if vars(args)[key] is not None:
            train_config[key] = vars(args)[key]

    data_config = json.load(open("data_config.json"))
    # also add the data config
    for key in data_config:
        if vars(args)[key] is None:
            train_config[key] = data_config[key]
    # print(train_config)

    main(train_config)
