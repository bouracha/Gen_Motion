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

# =============================================================================================================================================================================================
# =============================================================================================================================================================================================
#                     VDVAE
# =============================================================================================================================================================================================
# =============================================================================================================================================================================================

# ===============================================================
#                     Initial tests
# ===============================================================


#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vd_784_" --n_zs 784 500 300 200 50 10 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 500
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "ladder_2s_" --n_zs 2 2 2 2 2 2 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 500
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "ladder_2s_" --n_zs 2 2 2 2 2 2 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 501

# ===============================================================
#                     Generate
# ===============================================================

#python3 run_experiments.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vd_784_" --n_zs 784 500 300 200 50 10 2 --start_epoch 501 --icdf
#python3 run_experiments.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "ladder_2s_" --n_zs 2 2 2 2 2 2 2 --start_epoch 5001 --icdf

# ===============================================================
#                     Residual VDVAE
# ===============================================================

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vdvae_20_" --n_zs 20 16 10 8 5 3 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 501
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vdvae_2_" --n_zs 2 2 2 2 2 2 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vdvae_all20_" --n_zs 20 20 20 20 20 20 20 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vdvae_all5_" --n_zs 5 5 5 5 5 5 5 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000


# ===============================================================
#                     Generate Res VDVAE
# ===============================================================

#python3 run_experiments.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vdvae_2_" --n_zs 2 2 2 2 2 2 2 --start_epoch 1001 --icdf
#python3 run_experiments.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vdvae4_" --n_zs 2 4 6 8 10 12 14 --start_epoch 211 --icdf

# ===============================================================
#                     Residual VDVAE version 2
# ===============================================================

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "vdvae_first" --n_zs 14 12 10 8 6 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 10000 --start_epoch 5001
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "VDVAE_20_2_" --n_zs 20 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 731
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "VDVAE_2_2_" --n_zs 2 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "VDVAE_2_2_2" --n_zs 2 2 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "VDVAE_4_2" --n_zs 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 171

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "reverse3_" --n_zs 14 12 10 8 6 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 2011
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "reverse3_reproduce" --n_zs 14 12 10 8 6 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --warmup --warmup_block_length 20
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "reverse_" --n_zs 14 12 10 8 6 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 301
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "reverse_friday_" --n_zs 14 12 10 8 6 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1

# ===============================================================
#                     Warmup models no residuals
# ===============================================================

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "warmup_2_" --n_zs 2 --lr 0.001 --train_batch_size 1000 --n_epochs 100 --start_epoch 1
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "warmup_2_" --n_zs 2 --lr 0.001 --train_batch_size 1000 --n_epochs 100 --start_epoch 1 --warmup --warmup_block_length 50
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "warmup_2_2_" --n_zs 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1 --warmup --warmup_block_length 50
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "warmup_2_2_2_" --n_zs 2 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1 --warmup --warmup_block_length 50

# ===============================================================
#                     Warmup models VDVAE
# ===============================================================

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_2_" --n_zs 2 --lr 0.001 --train_batch_size 1000 --n_epochs 100 --start_epoch 1 --warmup --warmup_block_length 50
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_2_2_" --n_zs 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_2_2_2_" --n_zs 2 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 251 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_2_2_2_2_" --n_zs 2 2 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_12_10_8_6_4_2_" --n_zs 12 10 8 6 4 2 --lr 0.001 --train_batch_size 1000 --n_epochs 1000 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_12_10_8_6_4_2_" --n_zs 12 10 8 6 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 1000 --start_epoch 81 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_deeep_" --n_zs 12 10 8 6 5 4 3 2 --lr 0.001 --train_batch_size 1000 --n_epochs 1000 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_deeeeep_" --n_zs 32 16 12 10 8 6 5 4 3 2 --lr 0.001 --train_batch_size 1000 --n_epochs 1000 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_warmup_deeeeep_" --n_zs 32 16 12 10 8 6 5 4 3 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 2000 --start_epoch 361 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_64_" --n_zs 64 32 20 20 20 --lr 0.001 --train_batch_size 1000 --n_epochs 2000 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "rezero_warmup_64" --n_zs 64 32 16 8 4 --lr 0.001 --train_batch_size 1000 --n_epochs 1000 --start_epoch 1 --warmup --warmup_block_length 5

# ===============================================================
#                     VDVAE
# ===============================================================

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_2_" --n_zs 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 101
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_2_2_" --n_zs 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_2_2_2_" --n_zs 2 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_2_2_2_2_" --n_zs 2 2 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_12_10_8_6_4_2_" --n_zs 12 10 8 6 4 2 --lr 0.0001 --train_batch_size 1000 --n_epochs 1000 --start_epoch 1
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_12_10_8_6_4_2_" --n_zs 12 10 8 6 4 2 --lr 0.00003 --train_batch_size 1000 --n_epochs 1000 --start_epoch 191
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "res_12_10_8_6_4_2_" --n_zs 12 10 8 6 4 2 --lr 0.00001 --train_batch_size 1000 --n_epochs 1000 --start_epoch 241

# ===============================================================
#                     VDVAE with ReZero
# ===============================================================

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "rezero_2_2_2_2_" --n_zs 2 2 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1

# ===============================================================
#                     VDVAE with ReZero warmup
# ===============================================================

#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "rezero_warmup_2_2_2_2_" --n_zs 2 2 2 2 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "rezero_warmup_64" --n_zs 64 32 16 8 4 --lr 0.001 --train_batch_size 1000 --n_epochs 300 --start_epoch 1 --warmup --warmup_block_length 5
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "rezero_warmup-deteministic_64" --n_zs 64 32 16 8 4 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0.0
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "rezero_warmup-deteministic_64" --n_zs 64 32 16 8 4 --lr 0.0001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 941
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "rezero_warmup-deteministic_64_beta_lower" --n_zs 64 32 16 8 4 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0.0
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "ladder_" --n_zs 64 32 16 8 4 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0.0001
#python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "ladder_784_32" --n_zs 784 512 256 128 64 32 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0.0001 --warmup --highway_size 784

# ===============================================================
#                     VDVAE on pose
# ===============================================================

#python3 main.py --batch_norm --variational --name "pose" --n_zs 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1
#python3 main.py --batch_norm --variational --name "ladder_pose" --n_zs 8 4 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0.0001 --warmup
#python3 main.py --batch_norm --variational --name "ladder_output_variance" --n_zs 8 4 2 --lr 0.00001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0.001 --warmup_time 200 --beta_final 1.0 --output_variance
#python3 main.py --batch_norm --variational --name "ladder_beta" --n_zs 8 4 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0.0001 --warmup_time 20 --beta_final 0.001




#python3 main.py --motion --batch_norm --variational --name "motion_ae" --n_zs 4 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0 #.0001 #--warmup_time 200 --beta_final 0.001
#python3 main.py --motion --batch_norm --variational --name "motion_vae" --n_zs 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0 #.0001 #--warmup_time 200 --beta_final 0.001
#python3 main.py --motion --batch_norm --variational --name "temp" --n_zs 8 4 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0 #.0001 #--warmup_time 200 --beta_final 0.001
#python3 main.py --motion --timepoints 2 --batch_norm --variational --name "motion_ae_short" --n_zs 4 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --beta 0 #.0001 #--warmup_time 200 --beta_final 0.001

# ===============================================================
#                     VDVAE refactoring tests
# ===============================================================

python3 main.py --use_MNIST --use_bernoulli_loss --batch_norm --variational --name "refactor_test_" --n_zs 64 32 16 8 4 2 --lr 0.001 --train_batch_size 1000 --n_epochs 5000 --start_epoch 1 --warmup_time 20 --beta 0.0001