"""Common function to run online and offline algorithms."""
import time
import numpy as np


def run_algorithm(opt, max_nb_of_iter, verbose=0):
    """Run the algorithm setup with the defined optimizer.

    Parameters
    ----------
    opt: Optimizer Class
    max_nb_of_iter: int
        Maximum number of iteration
    verbose: int
        Verbosity level.

    Returns
    -------
    x_final: ndarray
        the estimated POGM solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    """
    start = time.perf_counter()
    cost_op = opt._cost_func
    # Perform the reconstruction
    if verbose > 0:
        print("Starting optimization...")
    opt.iterate(max_iter=max_nb_of_iter)
    end = time.perf_counter()
    if verbose > 0:
        if hasattr(cost_op, "cost"):
            print(" - final iteration number: ", cost_op._iteration)
            print(" - final log10 cost value: ", np.log10(cost_op.cost))
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)
    if hasattr(opt._grad, "linear_op"):
        x_final = opt._grad.linear_op.adj_op(opt.x_final)
    else:
        x_final = opt.x_final
    metrics = opt.metrics

    if hasattr(cost_op, "cost"):
        costs = cost_op._cost_list
    else:
        costs = None

    return x_final, costs, metrics


def run_online_algorithm(opt, kspace_generator, estimate_call_period, verbose=0):
    """Run online optimisation algorithm.

    At each step the obs_data is updated via the kspace_generator.

    Parameters
    ----------
    opt: instance of SetUp
        optimisation algorithm instance
    kspace_generator: instance of BaseKspaceGenerator
        The kspace_generator yielding the observed data to be updated.
    estimate_call_period: int, default None
        The period over which to retrieve an estimate of the online algorithm.
        If None, only the last estimate is retrieved.
    """
    opt.idx = 0
    cost_op = opt._cost_func
    # Perform the first reconstruction
    if verbose > 0:
        print("Starting optimization...")

    start = time.perf_counter()
    estimates = list()
    estimates += kspace_generator.opt_iterate(opt, estimate_call_period=estimate_call_period)

    end = time.perf_counter()
    if verbose > 0:
        if hasattr(cost_op, "cost"):
            print(" - final iteration number: ", cost_op._iteration)
            print(" - final cost value: ", cost_op.cost)
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)
    # Get the final solution
    observer_kwargs = opt.get_notify_observers_kwargs()

    ret_dict = dict()
    ret_dict['x_final'] = observer_kwargs['x_new']
    ret_dict['metrics'] = opt.metrics
    if hasattr(opt, '_y_new'):
        ret_dict['y_final'] = observer_kwargs['y_new']
    if hasattr(cost_op, "cost"):
        ret_dict['costs'] = cost_op._cost_list
    if estimates:
        ret_dict['x_estimates'] = estimates
    return ret_dict
