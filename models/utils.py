import torch

class AccumLoss(object):
    def __init__(self):
        self.reset()

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

def num_parameters_and_place_on_device(model):
    print(model)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(">>> total params: {:.2f}M".format(num_parameters / 1000000.0))
    if model.device == "cuda":
        print("Moving model to GPU")
        model.cuda()
    else:
        print("Using CPU")

def reparametisation_trick(mu, log_var, device):
    """

    :param mu: The mean of the latent variable to be formed (nbatch, n_z)
    :param log_var: The log variance of the latent variable to be formed (nbatch, n_z)
    :param device: CPU or GPU
    :return: latent variable (nbatch, n_z)
    """
    noise = torch.normal(mean=0, std=1.0, size=log_var.shape).to(torch.device(device))
    z = mu + torch.mul(torch.exp(log_var / 2.0), noise)

    return z

def kullback_leibler_divergence(mu_1, log_var_1, mu_2=None, log_var_2=None):
    """

    :param mu: The mean of the latent variable to be formed (nbatch, n_z)
    :param log_var: The log variance of the latent variable to be formed (nbatch, n_z)
    :return: gaussian analytical KL divergence for each datapoint averaged across the
    batch between two gaussian distributions p and q where by default q is N(0,1)
    """
    if mu_2 is None and log_var_2 is None:
        mu_2 = torch.zeros_like(mu_1)
        log_var_2 = torch.ones_like(log_var_1)
    KL_per_datapoint = 0.5 * torch.sum(-1 + log_var_2 - log_var_1 + torch.exp(log_var_1) + torch.pow((mu_1 - mu_2), 2)/(torch.exp(log_var_2)), axis=1)
    KL = torch.mean(KL_per_datapoint)

    return KL