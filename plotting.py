import matplotlib.pyplot as plt
import numpy as np


def plot_compare_Ys(ts, Ys1, Ys2):
    """Compare time series of Ys1, Ys2 and their delta.

    ts: times
    Ys1: first Y values
    Ys2: second Y values
    """
    assert len(ts) == len(Ys1) == len(Ys2), 'ts, Ys1 and Ys2 must all be same length.'
    plt.figure()
    plt.subplot(211)
    plt.title('Chaos with change in initial conditions')
    plt.plot(ts, Ys1)
    plt.plot(ts, Ys2)
    plt.ylabel('Y')

    plt.subplot(212)
    plt.plot(ts, Ys2 - Ys1)
    plt.ylabel('$\Delta$ Y')
    plt.xlabel('time')


def plot_lorenz63_figures(ts, Xs, Ys, Zs):
    """Plot figures as in L63.

    First figure: first 3001 (3x 1000) values of Ys plotted against time.
    Second figure: phase plot of 1400-1900 inc. of Y, Z.
                   includes some special values plotted as dots.

    ts: times
    Xs: X values
    Ys: Y values
    Zs: Z values
    """
    plt.figure()

    plt.subplot(311)
    plt.plot(range(0, 1001), Ys[0:1001])
    plt.ylim(-30, 30)
    plt.axhline(0, color='k')
    plt.xlabel('iteration')
    plt.ylabel('Y')

    plt.subplot(312)
    plt.plot(range(1000, 2001), Ys[1000:2001])
    plt.ylim(-30, 30)
    plt.axhline(0, color='k')
    plt.xlabel('iteration')
    plt.ylabel('Y')

    plt.subplot(313)
    plt.plot(range(2000, 3001), Ys[2000:3001])
    plt.ylim(-30, 30)
    plt.axhline(0, color='k')
    plt.xlabel('iteration')
    plt.ylabel('Y')

    plt.figure()
    plt.subplot(211)
    plt.plot(Ys[1400:1901], Zs[1400:1901])
    for i in range(1400, 1901, 100):
        plt.plot(Ys[i], Zs[i], 'bo')
        plt.annotate('{0:.0f}'.format(i / 100), (Ys[i], Zs[i]), xytext=(Ys[i] + 0.5, Zs[i] + 0.5))

    plt.plot(6 * np.sqrt(2), 27, 'ko')
    plt.annotate('C', (6 * np.sqrt(2), 27), xytext=(6 * np.sqrt(2) + 0.5, 27 + 0.5))
    plt.plot(-6 * np.sqrt(2), 27, 'ko')
    plt.annotate("C '", (-6 * np.sqrt(2), 27), xytext=(-6 * np.sqrt(2) + 0.5, 27 + 0.5))

    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.xlabel('Y')
    plt.ylabel('Z')

    plt.subplot(212)
    plt.plot(Ys[1400:1901], Xs[1400:1901])
    plt.ylim((20, -20))
    for i in range(1400, 1901, 100):
        plt.plot(Ys[i], Xs[i], 'bo')
        plt.annotate('{0:.0f}'.format(i / 100), (Ys[i], Xs[i]), xytext=(Ys[i] + 0.5, Xs[i] + 0.5))

    plt.plot(6 * np.sqrt(2), 6 * np.sqrt(2), 'ko')
    plt.annotate('C', (6 * np.sqrt(2), 6 * np.sqrt(2)), xytext=(6 * np.sqrt(2) + 0.5, 6 * np.sqrt(2) + 0.5))
    plt.plot(-6 * np.sqrt(2), -6 * np.sqrt(2), 'ko')
    plt.annotate("C '", (-6 * np.sqrt(2), -6 * np.sqrt(2)), xytext=(-6 * np.sqrt(2) + 0.5, -6 * np.sqrt(2) + 0.5))

    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.xlabel('Y')
    plt.ylabel('X')
