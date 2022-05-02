"""Online Gradient Operator."""

import numpy as np
from ...operators.gradient.gradient import GradAnalysis, GradSynthesis
from modopt.base.types import check_float, check_npndarray


class OnlineGradMixin:
    """A Mixin Class For Gradient Operator.

    Overide the obs_data setter and getter of GradBasic defined in Modopt.
    """

    @property
    def obs_data(self):
        """Observed data."""
        return self._obs_data

    @obs_data.setter
    def obs_data(self, data):
        if self._grad_data_type in (float, np.floating):
            data = check_float(data)
        check_npndarray(data, dtype=self._grad_data_type, writeable=True)

        self._obs_data = data

    # TODO: define a vector cost, with offline comparison if available.


class OnlineGradSynthesis(OnlineGradMixin, GradSynthesis):
    """
    Online gradient for Synthesis formulation.

    See Also:
    ---------
    GradSynthesis
    """

    pass


class OnlineGradAnalysis(OnlineGradMixin, GradAnalysis):
    """
    Online gradient for Analysis formulation.

    See Also:
    ---------
    GradAnalysis
    """

    pass
