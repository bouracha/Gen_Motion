#!/bin/bash
cd ..
# ===============================================================
#                     Train different models
# ===============================================================
#python3 main.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --name "deep_2" --output_variance --n_z 2 --lr 0.0001 --train_batch_size 1000
#python3 main.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --name "deep_2" --output_variance --n_z 2 --lr 0.0001 --train_batch_size 1000 --start_epoch 51 --n_epochs 300
#python3 main.py --hidden_layers 100 --lr 0.0001 --name "pca_ae"

# ===============================================================
#                     Degradation Experiments
# ===============================================================
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --name "deep_2_beta" --n_z 2 --degradation_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --name "deep_2" --output_variance --n_z 2 --degradation_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 111 --name "deep_2_wd" --n_z 2 --degradation_experiment
#python3 mapper.py --algorithm "PCA" --degradation_experiment --n_z 2

#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10_beta" --n_z 10 --degradation_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10" --output_variance --n_z 10 --degradation_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 131 --name "deep_10_wd" --n_z 10 --degradation_experiment
#python3 mapper.py --algorithm "PCA" --degradation_experiment --n_z 10

# ===============================================================
#                     Embedding Experiments
# ===============================================================
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 111 --name "deep_2_wd" --n_z 2 --embedding_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --name "deep_2" --output_variance --n_z 2 --embedding_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --name "deep_2_beta" --n_z 2 --embedding_experiment

#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 131 --name "deep_10_wd" --n_z 10 --embedding_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10" --output_variance --n_z 10 --embedding_experiment
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10_beta" --n_z 10 --embedding_experiment
# ===============================================================
#                     Generate ICDFs
# ===============================================================
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 111 --name "deep_2_wd" --n_z 2 --icdf
# ===============================================================
#                     Generate ICDFs
# ===============================================================
#python3 main.py --use_MNIST
#python3 main.py --use_MNIST --variational
#python3 main.py --use_MNIST --variational --name "20" --n_z 20
# ===============================================================
#                     Add various level of noise to inputs (model doesn't matter for just inputs)
# ===============================================================
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 111 --name "deep_2_wd" --n_z 2 --noise_to_inputs

#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 131 --name "deep_10_wd" --n_z 10 --noise_to_embeddings
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10" --output_variance --n_z 10 --noise_to_embeddings
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10_beta" --n_z 10 --noise_to_embeddings

#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 131 --name "deep_10_wd" --n_z 10 --de_noise
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10" --output_variance --n_z 10 --de_noise
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 171 --name "deep_10_beta" --n_z 10 --de_noise

#Smaller bottleneck
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --start_epoch 111 --name "deep_2_wd" --n_z 2 --noise_to_embeddings
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --name "deep_2" --output_variance --n_z 2 --noise_to_embeddings
#python3 run_experiments.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --start_epoch 111 --name "deep_2_beta" --n_z 2 --noise_to_embeddings
# ===============================================================
#                     Classifier train
# ===============================================================
#python3 classify.py --hidden_layers 100
#python3 classify.py --hidden_layers 100 --use_MNIST --n_epochs 100

#python3 classify.py --hidden_layers 100 --data_path "deep_2_wd_AE/embeddings/" --name "AE_2" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "inputs/" --name "inputs" --l2_reg --p_drop 0.5

#python3 classify.py --hidden_layers 100 --data_path "inputs/noise_0.0/" --name "inputs_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "inputs/noise_0.1/" --name "inputs_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "inputs/noise_0.3/" --name "inputs_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "inputs/noise_1.0/" --name "inputs_1.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "inputs/noise_3.0/" --name "inputs_3.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/embeddings/noise_0.0/" --name "VAE_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/embeddings/noise_0.1/" --name "VAE_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/embeddings/noise_0.3/" --name "VAE_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/embeddings/noise_1.0/" --name "VAE_1.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/embeddings/noise_0.0/" --name "AE_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/embeddings/noise_0.1/" --name "AE_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/embeddings/noise_0.3/" --name "AE_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/embeddings/noise_1.0/" --name "AE_1.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/embeddings/noise_0.0/" --name "beta_VAE_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/embeddings/noise_0.1/" --name "beta_VAE_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/embeddings/noise_0.3/" --name "beta_VAE_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/embeddings/noise_1.0/" --name "beta_VAE_1.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/recons/noise_0.0/" --name "VAE_recons_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/recons/noise_0.1/" --name "VAE_recons_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/recons/noise_0.3/" --name "VAE_recons_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_VAE/recons/noise_1.0/" --name "VAE_recons_1.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/recons/noise_0.0/" --name "VAE_beta_recons_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/recons/noise_0.1/" --name "VAE_beta_recons_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/recons/noise_0.3/" --name "VAE_beta_recons_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_beta_VAE/recons/noise_1.0/" --name "VAE_beta_recons_1.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/recons/noise_0.0/" --name "AE_recons_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/recons/noise_0.1/" --name "AE_recons_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/recons/noise_0.3/" --name "AE_recons_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_10_wd_AE/recons/noise_1.0/" --name "AE_recons_1.0" --l2_reg

#n_z = 2
#python3 classify.py --hidden_layers 100 --data_path "deep_2_VAE/embeddings/noise_0.0/" --name "2_VAE_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_VAE/embeddings/noise_0.1/" --name "2_VAE_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_VAE/embeddings/noise_0.3/" --name "2_VAE_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_VAE/embeddings/noise_1.0/" --name "2_VAE_1.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_VAE/embeddings/noise_3.0/" --name "2_VAE_3.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_2_wd_AE/embeddings/noise_0.0/" --name "2_AE_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_wd_AE/embeddings/noise_0.1/" --name "2_AE_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_wd_AE/embeddings/noise_0.3/" --name "2_AE_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_wd_AE/embeddings/noise_1.0/" --name "2_AE_1.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_wd_AE/embeddings/noise_3.0/" --name "2_AE_3.0" --l2_reg

#python3 classify.py --hidden_layers 100 --data_path "deep_2_beta_VAE/embeddings/noise_0.0/" --name "2_beta_VAE_0.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_beta_VAE/embeddings/noise_0.1/" --name "2_beta_VAE_0.1" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_beta_VAE/embeddings/noise_0.3/" --name "2_beta_VAE_0.3" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_beta_VAE/embeddings/noise_1.0/" --name "2_beta_VAE_1.0" --l2_reg
#python3 classify.py --hidden_layers 100 --data_path "deep_2_beta_VAE/embeddings/noise_3.0/" --name "2_beta_VAE_3.0" --l2_reg

# ===============================================================
#                     Classifier inference
# ===============================================================
#python3 classify.py --hidden_layers 100 --data_path "deep_2_wd_AE/embeddings/" --name "AE_2" --start_epoch 51 --inference
#python3 classify.py --hidden_layers 100 --data_path "inputs/" --name "inputs" --start_epoch 51 --inference

#python3 classify.py --hidden_layers 100 --start_epoch 21 --inference
#python3 classify.py --hidden_layers 100 --use_MNIST --n_epochs 100 --name "" --start_epoch 11 --inference

# ===============================================================
#                     Test runs
# ===============================================================

python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --name "test_" --n_z 20 --lr 0.0001 --train_batch_size 1000

#python3 main.py --batch_norm --hidden_layers 500 400 300 200 100 50 --variational --name "test_" --n_z 2 --lr 0.0001 --train_batch_size 1000 --beta 0.1
