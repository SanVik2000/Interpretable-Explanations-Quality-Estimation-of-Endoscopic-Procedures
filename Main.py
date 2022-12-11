import os
import torch
import argparse
from datetime import datetime
from torch._C import device

from Trainer import Trainer
from models.CNNLSTM import *
from models.ConvLSTM import *
from models.CNNTransformer import *
from Dataset import DataModule
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def add_args(parser):
    parser.add_argument("--test", action="store_true", default=False, help="test on test dataset")
    parser.add_argument("--vis", action="store_true", default=False, help="visualize attention matrices on the test set")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs per prune iteration")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--frame_count", type=int, default=100, help="frame count")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="number of data workers")
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed")
    parser.add_argument("--wandb", action="store_true", default=False, help="log metrics using wandb")
    parser.add_argument("--root_dir", type=str, default="", help="data root directory")
    parser.add_argument("--model_net", type=str, default="CNNLSTM", help="choose model from [CNNLSTM / ConvLSTM / CNNTrans / ConvTrans]")
    parser.add_argument("--head_type", type=str, default="pool", help="choose model from [pool / cls]")
    parser.add_argument("-o", "--out_dir", type=str, default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}", help="path to output directory [default: year-month-date_hour-minute]",)
    return parser

def Print_Line():
	STR = []
	Length = os.get_terminal_size().columns
	for i in range(0,Length):
		STR.append("=")
	STR = ''.join(STR)
	print(STR)
	return 0

def get_model(args):
    model_name = args.model_net
    model_dict = {'CNNLSTM': CNNLSTM(), 'CNNTrans' : CNNTrans(pool=args.head_type, vis=args.vis), 'ConvLSTM' : ConvLSTMModel()}
    return model_dict[model_name]

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def main():
    print("Binary Classification of Endoscopy Videos")

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    print("Arguments : " , args)

    if args.test:
        mode = 'test'
    else:
        mode = 'train'

    if args.vis:
        args.batch_size = 1
        
    model = get_model(args)

    print("Number of Parameters : ", count_parameters(model))
    model.to(device)
    
    if mode == 'train':
        dm = DataModule(mode=mode, args=args)
        dm.prepare_data()
        trainer = Trainer(model, dm, mode, args)
        trainer.fit(num_epochs=args.epochs)
    else:
        dm = DataModule(mode=mode, args=args)
        dm.prepare_data()
        trainer = Trainer(model, dm, mode, args)
        if args.vis:
            trainer.explain_model()
        else:
            acc = trainer.predict()
	

if __name__ == "__main__":
	main()
