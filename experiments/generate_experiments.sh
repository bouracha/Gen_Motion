#!/bin/bash

cd ..

#python3 generate.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --output_variance --name "deep_2" --grid_size 20
#python3 generate.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --name "deep_2_beta" --grid_size 20

#python3 generate.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 151 --name "deep_2_beta0.01" --grid_size 20
#python3 generate.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_2_beta0.01" --grid_size 20
#python3 generate.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 11 --name "deep_2_beta0.01" --grid_size 20

python3 generate.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 111 --name "deep_2_wd" --grid_size 20

