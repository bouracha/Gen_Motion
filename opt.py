import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--inference', dest='inference', action='store_true', help='include to run inference (classifier only)')
        self.parser.set_defaults(inference=False)
        self.parser.add_argument('--data_path', type=str, default="deep_2_wd_AE/embeddings/", help='Path to data- if applicable')
        # ===============================================================
        #                     Architecture options
        # ===============================================================
        self.parser.add_argument('--n_zs', nargs='+', type=int, default=[50, 10, 5, 2], help='input the out of distribution action')
        self.parser.add_argument('--hidden_layers', nargs='+', type=int, default=[500, 200, 100, 50], help='input the out of distribution action')
        self.parser.add_argument('--n_z', type=int, default=2, help='Number of latent variables')
        self.parser.add_argument('--variational', dest='variational', action='store_true', help='toggle VAE or AE')
        self.parser.set_defaults(variational=False)
        self.parser.add_argument('--output_variance', dest='output_variance', action='store_true', help='toggle model output variance or use as constant')
        self.parser.set_defaults(output_variance=False)
        self.parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='toggle use batch_norm or not')
        self.parser.set_defaults(batch_norm=False)
        self.parser.add_argument('--p_drop', type=float, default=0.0, help='dropout rate')
        # ===============================================================
        #                    Initialise options
        # ===============================================================
        self.parser.add_argument('--start_epoch', type=int, default=1, help='If not 1, load checkpoint at this epoch')
        self.parser.add_argument('--name', type=str, default="", help='Name of master folder containing model')
        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--beta', type=float, default=1.0, help='Downweighting of the KL divergence')
        self.parser.add_argument('--l2_reg', dest='l2_reg', action='store_true', help='toggle use l2 regularisation or not')
        self.parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train for')
        self.parser.add_argument('--train_batch_size', type=int, default=100, help='Number of epochs to train for')
        self.parser.add_argument('--test_batch_size', type=int, default=100, help='If not 1, load checkpoint at this epoch')
        self.parser.add_argument('--warmup', dest='warmup', action='store_true', help='toggle to use warmup for VDVAE')
        self.parser.set_defaults(warmup=False)
        self.parser.add_argument('--warmup_block_length', type=int, default=20, help='for how many epochs to warm up each latent variable')
        # ===============================================================
        #                     MNIST experiments options
        # ===============================================================
        self.parser.add_argument('--use_MNIST', dest='use_MNIST', action='store_true', help='toggle to use MNIST data instead')
        self.parser.set_defaults(use_MNIST=False)
        self.parser.add_argument('--use_bernoulli_loss', dest='use_bernoulli_loss', action='store_true', help='toggle to bernoulli of gauss loss')
        self.parser.set_defaults(use_bernoulli_loss=False)
        # ===============================================================
        #                     Experiments
        # ===============================================================
        self.parser.add_argument('--degradation_experiment', dest='degradation_experiment', action='store_true', help='toggle run degradation experiments')
        self.parser.set_defaults(degradation_experiment=False)
        self.parser.add_argument('--embedding_experiment', dest='embedding_experiment', action='store_true', help='toggle run embedding experiments')
        self.parser.set_defaults(embedding_experiment=False)
        self.parser.add_argument('--icdf', dest='icdf', action='store_true', help='toggle run ICDF generation')
        self.parser.set_defaults(icdf=False)
        self.parser.add_argument('--interpolate', dest='interpolate', action='store_true', help='toggle run interpolate experiment')
        self.parser.set_defaults(interpolate=False)
        self.parser.add_argument('--noise_to_inputs', dest='noise_to_inputs', action='store_true', help='add noise to inputs and save dataframes')
        self.parser.set_defaults(noise_to_inputs=False)
        self.parser.add_argument('--noise_to_embeddings', dest='noise_to_embeddings', action='store_true', help='add noise to inputs and save dataframes of embeddings')
        self.parser.set_defaults(noise_to_embeddings=False)
        self.parser.add_argument('--de_noise', dest='de_noise', action='store_true', help='add noise to inputs and save dataframes of embeddings')
        self.parser.set_defaults(de_noise=False)
        # ===============================================================
        #                     Experiment options
        # ===============================================================
        self.parser.add_argument('--grid_size', type=int, default=20, help='Size of grid in each dimension')


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt