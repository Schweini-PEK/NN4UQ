import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def ode_model(t, x, alpha):
    """An implementation of ODE model.

    :param t: The current timestamp.
    :param x: The current state variable vector, 1 * n.
    :param alpha: The random parameter.
    :return: a n * 1 vector.
    """
    dx_dt = np.zeros((1, len(x)))
    dx_dt[0, 0] = -alpha * x[0]
    return [np.transpose(dx_dt)]


def mlode_model(t, x, a):
    dx_dt = np.array([[1, -4], [4, -7]])
    return dx_dt


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
    ode_sol = solve_ivp(lambda t, x: ode_model(t, x, alpha), tspan, x0, method='LSODA')
    xs = ode_sol.y
    return xs.transpose()[-1, 0]


def multi_ode_predictor(x0, a, delta):
    """

    :param x0: The initial state variable.
    :param a: The
    :param delta: The whole time step Delta.
    :return: The state variable after delta.
    """
    # tspan = [0, delta]
    # ode_sol = solve_ivp(lambda t, x: mlode_model(t, x, a), tspan, x0, method='LSODA')
    # xs = ode_sol.y
    # return xs.transpose()[-1, 0]
    return np.dot(expm(delta * a), x0)
