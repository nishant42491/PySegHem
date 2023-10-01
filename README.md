**A Graphical Approach To Detect Brain Hemorrhage CT Scans**
# Semantic Segmentation Training Script

This script is designed for training the semantic segmentation model on CT scan data.

our preprint is available at: https://arxiv.org/abs/2202.06876

## Getting Started

### Prerequisites

Before running the script, ensure you have the following prerequisites:

- Python (3.x recommended)
- Required Python packages installed (see `requirements.txt`)

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/nishant42491/PySegHem.git
   ```

2. At the cloned directory run
   ```
   pip install -r requirements.txt
   ```

3. Run the PySegHem/train with the args given below:

The script accepts the following command-line arguments:

--path_to_train_dataset (str, required): Path to the directory containing the training dataset.

--apply_transforms (bool): Apply data augmentation transforms during training (optional).

--train_batch_size (int): Batch size for training.

--mean_for_train_data (float): Mean value used for data normalization during training.

--std_for_train_data (float): Standard deviation used for data normalization during training.

--val_batch_size (int): Batch size for validation.

--mean_for_val_data (float): Mean value used for data normalization during validation.

--std_for_val_data (float): Standard deviation used for data normalization during validation.

--path_to_val_dataset (str): Path to the directory containing the validation dataset.

--name_of_model (str): Name of the segmentation model to be used for training.

--path_for_directory_to_save_weights (str): Path to the directory to save model weights and logs.

--optimizer (str): The optimizer to use for training.

--learning_rate (float): The initial learning rate for the optimizer.

--loss_metric (str): The loss metric used for optimization.

--gpus (bool): Use GPUs for training if available.

--max_epochs (int): Maximum number of training epochs.

--wandb_login_key (str): WandB (Weights and Biases) login key for logging.

--wandb_project_name (str): Name of the WandB project for logging.

--wandb_runs_name (str): Name of the individual training run.



4. Example Usage
   ```
   python train.py --path_to_train_dataset /path/to/train/dataset --apply_transforms True --train_batch_size 32
               --mean_for_train_data 0.5 --std_for_train_data 0.2 --val_batch_size 16 --mean_for_val_data 0.5
               --std_for_val_data 0.2 --path_to_val_dataset /path/to/validation/dataset --name_of_model unet
               --path_for_directory_to_save_weights /path/to/save/weights --optimizer adam --learning_rate 0.001
               --loss_metric cross_entropy --gpus True --max_epochs 50 --wandb_login_key your_api_key
               --wandb_project_name your_project_name --wandb_runs_name experiment_1

   ```


