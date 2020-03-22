import numpy as np
import scipy.integrate
from scipy.linalg import expm


def ode_model(t, x, alpha):
    """An implementation of ODE model.

    :param t: The current timestamp.
    :param x: The current state variable vector, 1 * n.
    :param alpha: The random parameter.
    :return: a n * 1 vector.
    """
    dxdt = np.zeros((1, len(x)))
    dxdt[0, 0] = -alpha * x[0]
    return [np.transpose(dxdt)]


def ode_predictor(x0, alpha, delta):
    """Predict the state after dj as the length of the simulation time of the ODE model.

    :param x0: The initial state variable.
    :param alpha: The random parameter.
    :param delta: The time lag between two experiment measurements.
    :return: The next state calculated by the solver.
    """
    tspan = [0, delta]
    x0 = [x0]
    alpha = alpha
    ode_sol = scipy.integrate.solve_ivp(lambda t, x: ode_model(t, x, alpha), tspan, x0, method='LSODA')
    xs = ode_sol.y
    return xs.transpose()[-1, 0]


def multi_ode_predictor(x0, a, delta):
    """

    :param x0: The initial state variable.
    :param a: The
    :param delta: The whole time step Delta.
    :return: The state variable after delta.
    """
    return expm(delta * a) * x0
