"""Online reconstructor."""

import warnings
from modopt.opt.linear import Identity
from modopt.opt.proximity import IdentityProx
from ..operators.cost import SmartGenericCost

from ..optimizers.forward_backward import pogm_online, fista_online
from ..optimizers.primal_dual import condatvu_online
from ..optimizers.online_grad import gradient_online
from modopt.opt.algorithms import VanillaGenericGradOpt, AdaGenericGradOpt, RMSpropGradOpt,\
    MomentumGradOpt, ADAMOptGradOpt, SAGAOptGradOpt
from ..operators.gradient.online import OnlineGradAnalysis, OnlineGradSynthesis

OPTIMIZERS = {
    'condatvu': condatvu_online,
    'pogm': pogm_online,
    'fista': fista_online,
    'vanilla': lambda *args, **kwargs: gradient_online(VanillaGenericGradOpt, *args, **kwargs),
    'adagrad': lambda *args, **kwargs: gradient_online(AdaGenericGradOpt, *args, **kwargs),
    'rmsprop': lambda *args, **kwargs: gradient_online(RMSpropGradOpt, *args, **kwargs),
    'momentum': lambda *args, **kwargs: gradient_online(MomentumGradOpt, *args, **kwargs),
    'adam': lambda *args, **kwargs: gradient_online(ADAMOptGradOpt, *args, **kwargs),
    'saga': lambda *args, **kwargs: gradient_online(SAGAOptGradOpt, *args, **kwargs),
}

OPTIMIZERS_TYPE = {
    'condatvu': 'primal_dual',
    'pogm': 'forward_backward',
    'fista': 'forward_backward',
    }
ANALYSIS_OPT = {'condatvu': 'analysis',
                'vanilla-epoch': 'analysis',
                'momentum-epoch': 'analysis',
                }


class OnlineReconstructor:
    """
    This is the base reconstructor class for online reconstruction.

    This class holds somes parameters that are common for ALL MR Image reconstruction

    Notes
    -----
        For the Analysis case, finds the solution  for x of:
        ..math:: (1/2) * ||F x - y||^2_2 + mu * H (W x)
        For the Synthesis case, finds the solution of:
        ..math:: (1/2) * ||F Wt alpha - y||^2_2 + mu * H(alpha)

    Attributes
    ----------
    fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
    mri.operators
        Defines the fourier operator F.
    linear_op: object
        Defines the linear sparsifying operator W. This must operate on x and
        have 2 functions, op(x) and adj_op(coeff) which implements the
        operator and adjoint operator. For wavelets, this can be object of
        class WaveletN or WaveletUD2 from mri.operators
    prox_op: object, (optional default None)
        Defines the regularization operator for the regularization function H.
        If None, the  regularization chosen is Identity and the optimization
        turns to gradient descent.
    verbose: int
        verbosity level

    """

    def __init__(self, fourier_op, linear_op, regularizer_op=None, opt='condatvu', verbose=0):
        """Create OnlineReconstructor.

        Parameters
        ----------
        fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
        mri.operators
            Defines the fourier operator F.
        linear_op: object
            Defines the linear sparsifying operator W. This must operate on x and
            have 2 functions, op(x) and adj_op(coeff) which implements the
            operator and adjoint operator. For wavelets, this can be object of
            class WaveletN or WaveletUD2 from mri.operators
        regularizer_op: object, (optional default None)
            Defines the regularization operator for the regularization function H.
            If None, the  regularization chosen is Identity and the optimization
            turns to gradient descent.
        """
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        self.verbose = verbose
        if regularizer_op is None:
            warnings.warn("The prox_op is not set. Setting to identity. "
                          "Note that optimization is just a gradient descent.")
            self.prox_op = IdentityProx()
            self.linear_op = Identity()
        else:
            self.prox_op = regularizer_op
        assert opt in OPTIMIZERS.keys()
        self.opt = opt
        grad_formulation = ANALYSIS_OPT.get(opt, 'synthesis')
        if grad_formulation == 'analysis':
            self.gradient_op = OnlineGradAnalysis(self.fourier_op,
                                                  verbose=self.verbose,
                                                  num_check_lips=0,
                                                  lipschitz_cst=1.1)
        elif grad_formulation == 'synthesis':
            self.gradient_op = OnlineGradSynthesis(self.linear_op,
                                                   self.fourier_op,
                                                   verbose=self.verbose,
                                                   num_check_lips=0,
                                                   lipschitz_cst=1.1)
        else:
            raise RuntimeError("Unknown gradient formulation")
        self.grad_formulation = grad_formulation

    def reconstruct(self, kspace_gen, x_init=None, cost_op_kwargs=None, **kwargs):
        if cost_op_kwargs is None:
            cost_op_kwargs = dict()
        cost_op = SmartGenericCost(gradient_op=self.gradient_op,
                                   prox_op=self.prox_op,
                                   verbose=self.verbose >= 20,
                                   optimizer_type=OPTIMIZERS_TYPE.get(self.opt, 'forward_backward'),
                                   grad_formulation=self.grad_formulation,
                                   linear_op=self.linear_op,
                                   **cost_op_kwargs)

        return OPTIMIZERS[self.opt](
            kspace_generator=kspace_gen,
            gradient_op=self.gradient_op,
            linear_op=self.linear_op,
            prox_op=self.prox_op,
            cost_op=cost_op,
            x_init=x_init,
            verbose=self.verbose,
            **kwargs,)
