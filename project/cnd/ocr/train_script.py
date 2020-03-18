import argparse

import warnings
warnings.simplefilter("ignore")

from torch.utils.data import DataLoader
from Dataset.CropDataset import CropDataset, GeneratedDataset, CommonDataset
from model import CRNN
from config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
from transforms import get_transforms
from metrics import WrapCTCLoss, WrapAccuracy
from catalyst.dl import SupervisedRunner, CheckpointCallback
from CustomCallback import CustomCallback
import string
from pathlib import Path
import torch

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

# IF YOU USE GPU UNCOMMENT NEXT LINES:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

# define experiment path
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATHS = [
    Path(CV_CONFIG.get("data_path"))
]
# CHANGE YOUR BATCH SIZE
BATCH_SIZE = 100
# 400 EPOCH SHOULD BE ENOUGH
NUM_EPOCHS = 500
#NUM_EPOCHS = 25

#alphabet = " "
#alphabet += string.ascii_uppercase
#alphabet += "".join([str(i) for i in range(10)])

alphabet = 'ABEKMHOPCTYX' + '0123456789' + '-'

MODEL_PARAMS = {
    'image_height': 32,
    'number_input_channels': 1,
    'number_class_symbols': len(alphabet),
    'rnn_size': 64
}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("cuda enabled")
else:
    device = torch.device("cpu:0")
    print("cuda disabled")

if __name__ == "__main__":
    if EXPERIMENT_DIR.exists():
        print(f"Folder 'EXPERIMENT_DIR' already exists")
    else:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms(device)
    # define data path

    #dataset = CropDataset(transforms=transforms, cached=False)
    #dataset = GeneratedDataset(transforms = transforms, cached=False)
    dataset = CommonDataset(transforms = transforms, cached=False)
    train_dataset = dataset.Train # define your dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=6,
    )
    # IT IS BETTER TO SPLIT DATA INTO TRAIN|VAL AND USE METRICS ON VAL
    # val_dataset_paths = [p / "val" for p in DATASET_PATHS]
    val_dataset = dataset.Validate
    #
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = CRNN(**MODEL_PARAMS)
    # YOU CAN ADD CALLBACK IF IT NEEDED, FIND MORE IN
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # define callbacks if any
    callbacks = [
        CheckpointCallback(save_n_best=10),
        CustomCallback(metric_names=['accuracy'], meter_list=[WrapAccuracy(alphabet)])
    ]
    # input_keys - which key from dataloader we need to pass to the model
    runner = SupervisedRunner(input_key="image", input_target_key="targets")

    runner.train(
        model=model,
        criterion=WrapCTCLoss(alphabet),
        optimizer=optimizer,
        scheduler=scheduler,
        loaders={'train': train_loader, "valid": val_loader},
        logdir="./logs",
        num_epochs=NUM_EPOCHS,
        verbose=True,
        callbacks=callbacks
    )
