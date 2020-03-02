import torch
import torch.nn
import torch.nn.functional as F

from bbp_torch.modules.base_bayesian_module import BayesianModule

def kl_divergence_from_nn(model):

    """
    Gathers the KL Divergence from a nn.Module object
    Works by gathering each Bayesian layer kl divergence and summing it, doing nothing with the non Bayesian ones
    """
    kl_divergence = 0
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            kl_divergence += module.log_variational_posterior - module.log_prior
    return kl_divergence

