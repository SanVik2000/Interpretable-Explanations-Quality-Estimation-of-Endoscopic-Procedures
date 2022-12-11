#!/bin/bash
set -e

echo "CNN+LSTM"
python3 Main.py --root_dir /media/sanvik/Data/Dual_Degree_Project/ --model_net CNNLSTM --out_dir CNNLSTM

echo "Conv+LSTM"
python3 Main.py --root_dir /media/sanvik/Data/Dual_Degree_Project/ --model_net ConvLSTM --out_dir ConvLSTM

echo "CNN+Transformer+Pool"
python3 Main.py --root_dir /media/sanvik/Data/Dual_Degree_Project/ --model_net CNNTrans --out_dir CNNTransPool_4_128_Pool --head_type pool

echo "CNN+Transformer+CLS"
python3 Main.py --root_dir /media/sanvik/Data/Dual_Degree_Project/ --model_net CNNTrans --out_dir CNNTransPool_4_128_CLS --head_type cls

