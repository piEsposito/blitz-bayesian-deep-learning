import torch
import types

from blitz.modules.weight_sampler import GaussianVariational
from blitz.losses import kl_divergence_from_nn
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules import BayesianLSTM

def copy_func(f, name=None):
    '''
    return a function with same code, globals, defaults, closure, and 
    name (or provide a new name)
    '''
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
        f.__defaults__, f.__closure__)
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__) 
    return fn

def Flipout(nn_module):
    """
    Wrapper tha introduces flipout on the feedforwad operation of a BayesianModule layer in an non-intrusive way

    Parameters:
        nn_module: torch.nn.Module, BayesianModule -> Torch neural network module

    """
    class Flipout(nn_module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        if nn_module not in [BayesianLSTM, ]:
            def forward(self, x):
                outputs = super().forward_frozen(x)
                #getting the sign matrixes
                sign_input = x.clone().uniform_(-1, 1).sign()
                sign_output = outputs.clone().uniform_(-1, 1).sign()

                perturbed_outputs = super().forward(x * sign_input) * sign_output

                return outputs + perturbed_outputs
        else:
            def forward(self, x, states=None):
                outputs, states = super().forward_frozen(x, states)

                #getting the sign matrixes
                sign_input = x.clone().uniform_(-1, 1).sign()
                sign_output = outputs.clone().uniform_(-1, 1).sign()
                
                perturbed_outputs, perturbed_states = super().forward((x * sign_input), states) #* sign_output

                if not (type(states)==tuple):

                    return (perturbed_outputs + outputs, states + perturbed_states)
                
                return (perturbed_outputs + outputs, (states[i] + perturbed_states[i] for i in range(len(states))))

    return Flipout