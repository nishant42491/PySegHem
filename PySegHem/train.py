import torch
import argparse
from Data.create_data_loaders import create_train_loader,create_validation_loader
from models.PySegHem import PySegHem
from models.UNet import UNet

if __name__=='__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--path_to_train_dataset', action='store', type=str, required=True)
    my_parser.add_argument('--apply_transforms', action='store', type=bool)
    my_parser.add_argument('--train_batch_size', action='store', type=int)
    my_parser.add_argument('--val_batch_size', action='store', type=int)
    my_parser.add_argument('--path_to_val_dataset', action='store', type=str)
    my_parser.add_argument('--name_of_model', action='store', type=str)
    my_parser.add_argument('--path_for_directory_to_save_weights', action='store', type=str)
    my_parser.add_argument('--optimizer', action='store', type=str)
    my_parser.add_argument('--loss_metric', action='store', type=str)
    my_parser.add_argument('--wandb_login_key', action='store', type=str)
    my_parser.add_argument('--wandb_project_name', action='store', type=str)
    my_parser.add_argument('--wandb_runs_name', action='store', type=str)

    args = my_parser.parse_args()

    configs = {"train_path": args.path_to_train_dataset,
                "transform_flag": False if args.apply_transforms is None else args.apply_transforms,
                "train_batch": 8 if args.train_batch_size is None else args.train_batch_size,
                "val_batch_size": 8 if args.val_batch_size is None else args,
                'path_to_val_dataset': args.path_to_val_dataset,
                'model': 'PySegHem' if args.name_of_model is None else args.name_of_model,
                'save_directory': "./weights" if args.path_for_directory_to_save_weights is None else
                args.path_for_directory_to_save_weights,
                'optimizer': "Adam" if args.optimizer is None else args.optimizer,
                'loss_metric': 'dice_loss' if args.loss_metric is None else args.loss_metric,
                'wandb_key': args.wandb_login_key,
                'wandb_project_name': args.wandb_project_name,
                'wandb_runs_name': args.wandb_runs_name, }




