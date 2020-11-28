import torch
import types

from blitz.modules.weight_sampler import TrainableRandomDistribution
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
    Wrapper tha introduces flipout on the feedforwad operation of a BayesianModule layer in an non-intrusive way as in
    @misc{wen2018flipout,
        title={Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches},
        author={Yeming Wen and Paul Vicol and Jimmy Ba and Dustin Tran and Roger Grosse},
        year={2018},
        eprint={1803.04386},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    Parameters:
        nn_module: torch.nn.Module, BayesianModule -> Torch neural network module
    """

    class Flipout(nn_module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        if nn_module not in [BayesianLSTM, ]:
            def forward(self, x):
                outputs = super().forward_frozen(x)
                # getting the sign matrixes
                sign_input = x.clone().uniform_(-1, 1).sign()
                sign_output = outputs.clone().uniform_(-1, 1).sign()

                perturbed_outputs = super().forward(x * sign_input) * sign_output

                return outputs + perturbed_outputs
        else:
            def forward(self, x, states=None):
                outputs, states = super().forward_frozen(x, states)

                # getting the sign matrixes
                sign_input = x.clone().uniform_(-1, 1).sign()
                sign_output = outputs.clone().uniform_(-1, 1).sign()

                perturbed_outputs, perturbed_states = super().forward((x * sign_input), states)  # * sign_output

                if not (type(states) == tuple):
                    return (perturbed_outputs + outputs, states + perturbed_states)

                return (perturbed_outputs + outputs, (states[i] + perturbed_states[i] for i in range(len(states))))

    return Flipout


def Radial(nn_module):
    """
    Wrapper tha introduces the Radial feature on the feedforwad operation of a BayesianModule layer in an non-intrusive way as in
    @misc{farquhar2019radial,
        title={Radial Bayesian Neural Networks: Beyond Discrete Support In Large-Scale Bayesian Deep Learning},
        author={Sebastian Farquhar and Michael Osborne and Yarin Gal},
        year={2019},
        eprint={1907.00865},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }
    Parameters:
        nn_module: torch.nn.Module, BayesianModule -> Torch neural network module
    """

    def sample_radial(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is
        a function from a trainable parameter, and adding a mean sets those weights as the current ones
        We divide the random parameter per its norm to perform radial bnn inference
        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * (self.eps_weight / torch.norm(self.eps_weight))
        return self.w

    setattr(nn_module, "sample", sample_radial)
    return nn_module