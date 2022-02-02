import os
import pickle
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
from Data.create_data_loaders import create_train_loader,create_validation_loader
from models.PySegHem import PySegHem
from models.UNet import UNet
from torch.optim import Adam, AdamW, SGD, RMSprop
from metrics.dice_loss import Dice_Loss
from metrics.dice_bce_loss import Dice_BCE_Loss
from metrics.focal_tversky import Tversky_Focal_Loss
from metrics.iou_accuracy import iou_accuracy
from metrics.accuracy import accuracy


class New_Model(pl.LightningModule):
    def __init__(self, model_name, l_fn, opt, lrn):
        super(New_Model, self).__init__()
        self.model = model_name()
        self.loss_fn = l_fn
        self.opt = opt
        self.lrn = lrn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        outs = self(x)

        loss1 = self.loss_fn(outs, y)

        self.log("training monitoring metric", loss1)

        return loss1

    def validation_step(self, batch, batch_idx):
        x, y = batch

        outs = self(x)

        loss1 = self.loss_fn(outs, y)

        self.log("Validation monitoring metric", loss1)

        return loss1

    def configure_optimizers(self):
        return self.opt(self.parameters(), lr=self.lrn)




if __name__=='__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--path_to_train_dataset', action='store', type=str, required=True)
    my_parser.add_argument('--apply_transforms', action='store', type=bool)
    my_parser.add_argument('--train_batch_size', action='store', type=int)
    my_parser.add_argument('--mean_for_train_data', action='store', type=float)
    my_parser.add_argument('--std_for_train_data', action='store', type=float)
    my_parser.add_argument('--val_batch_size', action='store', type=int)
    my_parser.add_argument('--mean_for_val_data', action='store', type=float)
    my_parser.add_argument('--std_for_val_data', action='store', type=float)
    my_parser.add_argument('--path_to_val_dataset', action='store', type=str)
    my_parser.add_argument('--name_of_model', action='store', type=str)
    my_parser.add_argument('--path_for_directory_to_save_weights', action='store', type=str)
    my_parser.add_argument('--optimizer', action='store', type=str)
    my_parser.add_argument('--learning_rate', action='store', type=float)
    my_parser.add_argument('--loss_metric', action='store', type=str)
    my_parser.add_argument('--gpus', action='store', type=bool)
    my_parser.add_argument('--max_epochs', action='store', type=int)
    my_parser.add_argument('--wandb_login_key', action='store', type=str)
    my_parser.add_argument('--wandb_project_name', action='store', type=str)
    my_parser.add_argument('--wandb_runs_name', action='store', type=str)

    args = my_parser.parse_args()

    configs = {"train_path": args.path_to_train_dataset,
               "transform_flag": False if args.apply_transforms is None else args.apply_transforms,
               "train_batch": 8 if args.train_batch_size is None else args.train_batch_size,
               "train_mean": 0.2289 if args.mean_for_train_data is None else args.mean_for_train_data,
               "train_std": 0.312 if args.std_for_train_data is None else args.std_for_train_data,
               "val_mean": 0.2289 if args.mean_for_val_data is None else args.mean_for_val_data,
               'val_std':  0.312 if args.std_for_val_data is None else args.std_for_val_data,
               "val_batch_size": 8 if args.val_batch_size is None else args,
               'path_to_val_dataset': args.path_to_val_dataset,
               'model': 'PySegHem' if args.name_of_model is None else args.name_of_model,
               'max_epochs': 50 if args.max_epochs is None else args.max_epochs,
               'gpus': False if args.gpus is None else args.gpus,
               'save_directory': "./weights" if args.path_for_directory_to_save_weights is None else
               args.path_for_directory_to_save_weights,
               'optimizer': "Adam" if args.optimizer is None else args.optimizer,
               'loss_metric': 'dice' if args.loss_metric is None else args.loss_metric,
               'learning_rate': 0.0001 if args.learning_rate is None else args.learning_rate,
               'wandb_key': args.wandb_login_key,
               'wandb_project_name': args.wandb_project_name,
               'wandb_runs_name': args.wandb_runs_name, }

    model_dict = {"PySegHem": PySegHem, "UNet": UNet, }
    optim_dict = {"Adam": Adam,
                  "AdamW": AdamW,
                  "RMSProp": RMSprop,
                  "SGD": SGD,
                  }

    loss_fn_dict = {"dice": Dice_Loss,
                    "dice_bce": Dice_BCE_Loss,
                    "accuracy": accuracy,
                    "focal_tversky": Tversky_Focal_Loss,
                    "iou": iou_accuracy, }

    if configs['save_directory'] == "./weights":
        os.makedirs("./weights", exist_ok=True)

    try:
        curr_model=model_dict[configs['model']]

    except:
        print("ERROR Enter Valid Model Name!!!! Check Documentation for valid model names...")
        curr_model = None


    try:
        curr_optim = optim_dict[configs['optimizer']]

    except:
        print("Error Enter valid Optimizer name check docs for valid optimizer names")
        curr_optim = None

    try:
        curr_loss_fn = loss_fn_dict[configs['loss_metric']]

    except:
        print("Error enter valid loss function name")
        curr_loss_fn = None

    train_loader = create_train_loader(path_to_train_data=configs['train_path'], batch_size=configs["train_batch"],
                                       mean_data=configs["train_mean"], std_data=configs["train_std"])

    if configs['path_to_val_dataset'] is not None:
        val_loader = create_validation_loader(path_to_validation_data=configs['path_to_val_dataset'],
                                              batch_size=configs['val_batch_size'], mean_data=configs['val_mean'],
                                              std_data=configs['val_std'])

    else:
        val_loader = None





    checkpoint_callback = ModelCheckpoint(
        monitor='Validation monitoring metric',
        dirpath=configs['save_directory'],
        save_last=True,
        save_top_k=5,
        filename='CT_Model-{epoch:02d}-{val_loss:.2f}',
        mode='min',
        every_n_epochs=4,)

    if configs['wandb_key'] is not None:
        wandb.login(key=configs['wandb_key'], relogin=True)
        wandb_logger = WandbLogger(project=configs['wandb_project_name'], name=configs['wandb_project_name'])

        if configs['gpus']:
            trainer = Trainer(default_root_dir=configs['save_directory'], callbacks=[checkpoint_callback,],
                              max_epochs=configs['max_epochs'], gpus=1, logger=wandb_logger, enable_progress_bar=True,
                              enable_model_summary=True, )
        else:
            trainer = Trainer(default_root_dir=configs['save_directory'], callbacks=[checkpoint_callback, ],
                              max_epochs=configs['max_epochs'],  logger=wandb_logger, enable_progress_bar=True,
                              enable_model_summary=True, )

    else:
        if configs['gpus']:
            trainer = Trainer(default_root_dir=configs['save_directory'], callbacks=[checkpoint_callback,],
                              max_epochs=configs['max_epochs'], gpus=1, enable_progress_bar=True,
                              enable_model_summary=True)
        else:
            trainer = Trainer(default_root_dir=configs['save_directory'], callbacks=[checkpoint_callback, ],
                              max_epochs=configs['max_epochs'], enable_progress_bar=True, enable_model_summary=True)

    ct_model = New_Model(curr_model, curr_loss_fn, curr_optim, configs['learning_rate'])
    with open(os.path.join(configs['save_directory'],"saved_model.pkl"),'wb') as fobj:
        pickle.dump(ct_model,fobj)
        fobj.close()

    trainer.fit(model=ct_model, train_dataloader=train_loader, val_dataloaders=val_loader)













