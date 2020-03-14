import numpy as np
import scipy.integrate


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


def state_predictor(x0, alpha, dj):
    """Predict the state after dj as the length of the simulation time of the ODE model.

    :param x0: The initial state variable.
    :param alpha: The random parameter.
    :param dj: The time lag between two experiment measurements.
    :return: ???
    """
    tspan = [0, dj]
    x0 = [x0]
    alpha = alpha
    ode_sol = scipy.integrate.solve_ivp(lambda t, x: ode_model(t, x, alpha), tspan, x0, method='LSODA')
    # LSODA is the most consistent solver
    xs = ode_sol.y
    return xs.transpose()[-1, 0]


def trajectory(xinit, tinit, tfinal, alpha):
    """Predict the state after dj as the length of the simulation time of the ODE model.

    :param x0: The initial state variable.
    :param alpha: The random parameter.
    :param dj: The time lag between two experiment measurements.
    :return: ???
    """
    tspan = [0, tfinal]
    x0 = [xinit]
    alpha = alpha
    ode_sol = scipy.integrate.solve_ivp(lambda t, x: ode_model(t, x, alpha), tspan, x0, method='LSODA')
    # LSODA is the most consistent solver
    ts = ode_sol.t
    xs = ode_sol.y
    return ts, xs
