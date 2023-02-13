"""Implementation of Deterministic Nonperiodic Flow (aka chaos) as in Lorenz 1963 (L63).

DOI: https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2
"""

import numpy as np
import pylab as plt
from scipy.signal import argrelextrema

from plotting import plot_lorenz63_figures, plot_compare_Ys


def _lgp_format_number(x):
    """Formats a number in LGP-30 style"""
    if x >= 0:
        return '{0:04.0f}'.format(x * 10)
    else:
        return '-{0:04.0f}'.format(abs(x) * 10)


def lgp_30_output(Xs, Ys, Zs, skip=5, length=33, indices=None):
    """Outputs as in L63: multiplied by 10 with right of dp discarded
    N.B. rounding is used here, not sure it is used in orig.

    LGP-30 computer: https://en.wikipedia.org/wiki/LGP-30
    This is a 31-bit computer(!) so hard to emulate exactly.

    prints values to stdout.

    Xs: X values
    Ys: Y values
    Zs: Z values
    skip: how many values to skip over
    length: total number of values to print
    indices: only output requesting indices

    returns: None
    """
    f_ = _lgp_format_number
    if indices is None:
        indices = range(0, skip * length, skip)
    fmt_string = '{0:04d} {1: >6} {2: >6} {3: >6}'
    print('')
    print('{0: <4} {1: <6} {2: <6} {3: <6}'.format(' N', '   X', '   Y', '   Z'))
    for i in indices:
        X, Y, Z = Xs[i], Ys[i], Zs[i]
        print(fmt_string.format(i, f_(X), f_(Y), f_(Z)))


def fX(X, Y, Z, sigma, r, b):
    """L63 Eqn. 25 RHS"""
    return -sigma * X + sigma * Y


def fY(X, Y, Z, sigma, r, b):
    """L63 Eqn. 26 RHS"""
    return -X * Z + r * X - Y


def fZ(X, Y, Z, sigma, r, b):
    """L63 Eqn. 27 RHS"""
    return X * Y - b * Z


class Lorenz63:
    """Implementation of Lorenz 63 with various solvers."""

    schemes = {
        'da': 'double approx. (used by Lorenz)',
        'ft': 'forward time',
        'rk4': 'Runge-Kutta 4th order',
    }

    def __init__(self, nt, dt, X0=0, Y0=1, Z0=0, sigma=10, r=28, b=8 / 3, scheme='da'):
        """Initialize the class with the given settings.

        nt: number of timesteps
        dt: delta t between timesteps
        X0: initial X value
        Y0: initial Y value
        Z0: initial Z value
        sigma: parameter as in L63
        r: parameter as in L63
        b: parameter as in L63
        scheme: one of 'da', 'ft', 'rk4' (double approx., forward time, and Runge-Kutta-4th order)
        """
        if scheme not in self.schemes:
            raise ValueError(f'Scheme "{scheme}" not recognized\nMust be one of: {", ".join(self.schemes.keys())}')

        self.nt = nt
        self.dt = dt
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0
        self.sigma = sigma
        self.r = r
        self.b = b
        self.scheme = scheme

    def run(self):
        """Run the model with the current settings.

        returns: (times, Xs, Ys, Zs)
        """
        scheme = self.scheme
        nt = self.nt
        dt = self.dt
        sigma = self.sigma
        r = self.r
        b = self.b

        Xs, Ys, Zs = [self.X0], [self.Y0], [self.Z0]
        X, Y, Z = self.X0, self.Y0, self.Z0
        self.times = np.linspace(0, nt * dt, nt)
        for t in self.times[1:]:
            if scheme == 'rk4':
                # RK4.
                kX1 = fX(X, Y, Z, sigma, r, b)
                kY1 = fY(X, Y, Z, sigma, r, b)
                kZ1 = fZ(X, Y, Z, sigma, r, b)

                kX2 = fX(X + dt / 2 * kX1, Y + dt / 2 * kY1, Z + dt / 2 * kZ1, sigma, r, b)
                kY2 = fY(X + dt / 2 * kX1, Y + dt / 2 * kY1, Z + dt / 2 * kZ1, sigma, r, b)
                kZ2 = fZ(X + dt / 2 * kX1, Y + dt / 2 * kY1, Z + dt / 2 * kZ1, sigma, r, b)

                kX3 = fX(X + dt / 2 * kX2, Y + dt / 2 * kY1, Z + dt / 2 * kZ1, sigma, r, b)
                kY3 = fY(X + dt / 2 * kX2, Y + dt / 2 * kY1, Z + dt / 2 * kZ1, sigma, r, b)
                kZ3 = fZ(X + dt / 2 * kX2, Y + dt / 2 * kY1, Z + dt / 2 * kZ1, sigma, r, b)

                kX4 = fX(X + dt * kX2, Y + dt * kY1, Z + dt * kZ1, sigma, r, b)
                kY4 = fY(X + dt * kX2, Y + dt * kY1, Z + dt * kZ1, sigma, r, b)
                kZ4 = fZ(X + dt * kX2, Y + dt * kY1, Z + dt * kZ1, sigma, r, b)

                X = X + dt / 6 * (kX1 + 2 * kX2 + 2 * kX3 + kX4)
                Y = Y + dt / 6 * (kY1 + 2 * kY2 + 2 * kY3 + kY4)
                Z = Z + dt / 6 * (kZ1 + 2 * kZ2 + 2 * kZ3 + kZ4)
            else:
                X1 = X + dt * fX(X, Y, Z, sigma, r, b)
                Y1 = Y + dt * fY(X, Y, Z, sigma, r, b)
                Z1 = Z + dt * fZ(X, Y, Z, sigma, r, b)
                if scheme == 'da':
                    X2 = X1 + dt * fX(X1, Y1, Z1, sigma, r, b)
                    Y2 = Y1 + dt * fY(X1, Y1, Z1, sigma, r, b)
                    Z2 = Z1 + dt * fZ(X1, Y1, Z1, sigma, r, b)

                    X = 0.5 * (X + X2)
                    Y = 0.5 * (Y + Y2)
                    Z = 0.5 * (Z + Z2)
                else:
                    X, Y, Z = X1, Y1, Z1

            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)

        self.Xs = np.array(Xs)
        self.Ys = np.array(Ys)
        self.Zs = np.array(Zs)

        return self.times, self.Xs, self.Ys, self.Zs


def lorenz63_output(scheme='da'):
    """Output similar tables/plots to Lorenz 63 paper.

    da: double approx (what Lorenz used)"""
    settings = dict(X0=0, Y0=1, Z0=0, sigma=10, b=8 / 3, r=28, scheme=scheme)

    ts, Xs, Ys, Zs = lorenz_solver(6000, 0.01, **settings)

    # print_equations()
    lgp_30_output(Xs, Ys, Zs)
    Z_extrema_indices = argrelextrema(Zs, np.greater)[0]
    lgp_30_output(Xs, Ys, Zs, indices=Z_extrema_indices)

    plot_lorenz63_figures(ts, Xs, Ys, Zs)


def chaos_demo():
    """Demonstrate chatoic behaviour when initial conditions are changed.

    Settings as in L63.
    """
    settings = dict(X0=0, Y0=1, Z0=0, sigma=10, b=8 / 3, r=28, scheme='da')

    model = Lorenz63(6000, 0.01, **settings)
    ts, Xs1, Ys1, Zs1 = model.run()

    print('Output similar to that in Lorenz 63')
    print('-----------------------------------')
    lgp_30_output(Xs1, Ys1, Zs1)
    Z_extrema_indices = argrelextrema(Zs1, np.greater)[0]
    lgp_30_output(Xs1, Ys1, Zs1, indices=Z_extrema_indices)
    plot_lorenz63_figures(ts, Xs1, Ys1, Zs1)

    # slightly different ICs.
    settings['Y0'] = 1 + 1e-9
    model = Lorenz63(6000, 0.01, **settings)
    ts, Xs2, Ys2, Zs2 = model.run()

    plot_compare_Ys(ts, Ys1, Ys2)


if __name__ == '__main__':
    chaos_demo()
    plt.show()
