import torch
from blitz.losses import kl_divergence_from_nn
from blitz.modules.base_bayesian_module import BayesianModule
def variational_estimator(nn_class):

    def nn_kl_divergence(self):
        #returns the kl_divergence of the model, works for better encapsulation
        #nothing new under the sun
        return kl_divergence_from_nn(self)
    
    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo(self,
                    inputs,
                    labels,
                    criterion,
                    sample_nbr):

        #samples the ELBO loss for a batch of "datapoint, input" pars
        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss += self.nn_kl_divergence()
        return loss / sample_nbr
    
    setattr(nn_class, "sample_elbo", sample_elbo)


    def freeze_model(self):
        #frezes all the BayesianModule's on the current model
        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                module.freeze = True

    setattr(nn_class, "freeze", freeze_model)

    return nn_class