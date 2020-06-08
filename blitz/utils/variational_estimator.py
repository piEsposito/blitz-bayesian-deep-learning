import torch

from blitz.modules.weight_sampler import GaussianVariational
from blitz.losses import kl_divergence_from_nn
from blitz.modules.base_bayesian_module import BayesianModule, BayesianRNN

def variational_estimator(nn_class):
    """
    This decorator adds some util methods to a nn.Module, in order to facilitate the handling of Bayesian Deep Learning features

    Parameters:
        nn_class: torch.nn.Module -> Torch neural network module

    Returns a nn.Module with methods for:
        (1) Gathering the KL Divergence along its BayesianModules;
        (2) Sample the Elbo Loss along its variational inferences (helps training)
        (3) Freeze the model, in order to predict using only their weight distribution means
        (4) Specifying the variational parameters by using some prior weights after training the NN as a deterministic model
    """

    def nn_kl_divergence(self):
        """Returns the sum of the KL divergence of each of the BayesianModules of the model, which are from
            their posterior current distribution of weights relative to a scale-mixtured prior (and simpler) distribution of weights

            Parameters:
                N/a

            Returns torch.tensor with 0 dim.      
        
        """
        return kl_divergence_from_nn(self)
    
    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo(self,
                    inputs,
                    labels,
                    criterion,
                    sample_nbr,
                    complexity_cost_weight=1):

        """ Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels

                The ELBO Loss consists of the sum of the KL Divergence of the model 
                 (explained above, interpreted as a "complexity part" of the loss)
                 with the actual criterion - (loss function) of optimization of our model
                 (the performance part of the loss). 

                As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                 samples of the weights in order to gather a better approximation for the loss.

            Parameters:
                inputs: torch.tensor -> the input data to the model
                labels: torch.tensor -> label data for the performance-part of the loss calculation
                        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
                criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather
                            the performance cost for the model
                sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to 
                            gather the loss to be .backwarded in the optimization of the model.        
        
        """

        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            loss += criterion(outputs, labels) 
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return loss / sample_nbr
    
    setattr(nn_class, "sample_elbo", sample_elbo)


    def freeze_model(self):
        """
        Freezes the model by making it predict using only the expected value to their BayesianModules' weights distributions
        """
        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                module.freeze = True

    setattr(nn_class, "freeze_", freeze_model)

    def unfreeze_model(self):
        """
        Unfreezes the model by letting it draw its weights with uncertanity from their correspondent distributions
        """
        
        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                module.freeze = False

    setattr(nn_class, "unfreeze_", unfreeze_model)

    def moped(self, delta=0.1):
        """
        Sets the sigma for the posterior distribution to delta * mu as proposed in

        @misc{krishnan2019specifying,
            title={Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes},
            author={Ranganath Krishnan and Mahesh Subedar and Omesh Tickoo},
            year={2019},
            eprint={1906.05323},
            archivePrefix={arXiv},
            primaryClass={cs.NE}
        }   


        """
        for module in self.modules():
            if isinstance(module, (BayesianModule)):

                for attr in module.modules():
                    if isinstance(attr, (GaussianVariational)):
                        attr.rho.data = torch.log(torch.expm1(delta * torch.abs(attr.mu.data) ) + 1e-10)
        self.unfreeze_()

    setattr(nn_class, 'MOPED_', moped)

    def mfvi_forward(self, inputs, sample_nbr=10):
        """
        Performs mean-field variational inference for the variational estimator model:
            Performs sample_nbr forward passes with uncertainty on the weights, returning its mean and standard deviation

        Parameters:
            inputs: torch.tensor -> the input data to the model
            sample_nbr: int -> number of forward passes to be done on the data
        Returns:
            mean_: torch.tensor -> mean of the perdictions along each of the features of each datapoint on the batch axis
            std_: torch.tensor -> std of the predictions along each of the features of each datapoint on the batch axis


        """
        result = torch.stack([self(inputs) for _ in range(sample_nbr)])
        return result.mean(dim=0), result.std(dim=0)
    
    setattr(nn_class, 'mfvi_forward', mfvi_forward)
    
    def forward_with_sharpening(self, x, labels, criterion):
        preds = self(x)
        loss = criterion(preds, labels)
        
        for module in self.modules():
            if isinstance(module, (BayesianRNN)):
                module.loss_to_sharpen = loss
                
        y_hat: self(x)
        
        for module in self.modules():
            if isinstance(module, (BayesianRNN)):
                module.loss_to_sharpen = None
                
        return self(x,)
        
    setattr(nn_class, 'forward_with_sharpening', forward_with_sharpening)
        
        
        

    return nn_class
