import os
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--num_trials', type=int, default='10',
                    help='number of trials of randomly selected hyperparameters')
parser.add_argument('--variational', type=bool, default=False, help='true if want to include a latent variable')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')

opt = parser.parse_args()


def random_hyperparameters():
    n_z_values = [8, 16, 32]
    num_decoder_stage_values = [1, 2, 3, 4, 5, 6]
    alpha_log_range = (-5.298317366548036, -9.903487552536127)  # (0.005, 0.0005)
    lambda_log_range = (2.302585092994046, -13.815510557964274)  # (10, 0.00001)
    dropout_range = (0.0, 0.9)

    n_z = random.choice(n_z_values)
    num_decoder_stage = random.choice(num_decoder_stage_values)
    alpha = np.e ** (random.uniform(alpha_log_range[0], alpha_log_range[1]))
    lambda_ = np.e ** (random.uniform(lambda_log_range[0], lambda_log_range[1]))
    dropout = random.uniform(dropout_range[0], dropout_range[1])

    return n_z, num_decoder_stage, alpha, lambda_, dropout


for i in range(opt.num_trials):
    (n_z, num_decoder_stage, lr, lambda_, dropout) = random_hyperparameters()
    command = 'python3 main.py --lr ' + str(lr) + ' --lambda ' + str(lambda_) + ' --n_z ' + str(
        n_z) + ' --num_decoder_stage ' + str(num_decoder_stage) + ' --variational ' + str(
        opt.variational) + ' --lr_gamma 1.0 --epoch ' + str(
        opt.epoch) + ' --dropout ' + str(dropout) + ' --input_n 10 --output 10 --dct_n 20 --data_dir h3.6m/dataset/ --out_of_distribution walking'
    os.system(command)
