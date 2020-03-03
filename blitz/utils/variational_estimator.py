import torch
from blitz.losses import kl_divergence_from_nn

def variational_estimator(nn_class):

    def nn_kl_divergence(self):
        return kl_divergence_from_nn(self)

    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)
    return nn_class