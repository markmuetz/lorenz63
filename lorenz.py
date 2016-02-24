'Implementation of Deterministic Nonperiodic Flow (aka chaos) as in Lorenz 1963'
from __future__ import division

import numpy as np
import pylab as plt
from scipy.signal import argrelextrema

from lorenz_symbolic import print_equations


def lgp_format_number(x):
    if x >= 0:
	return '{0:04.0f}'.format(x * 10)
    else:
	return '-{0:04.0f}'.format(abs(x) * 10)


def lgp_30_output(Xs, Ys, Zs, skip=5, length=33, indices=None):
    '''Outputs as in in paper: multiplied by 10 with right of dp discarded
    N.B. rounding is used here, not sure it is used in orig.

    LGP-30 computer: https://en.wikipedia.org/wiki/LGP-30
    This is a 31-bit computer(!) so hard to emulate exactly.
    '''
    f_ = lgp_format_number
    if indices is None:
	indices = range(0, skip*length, skip)
    fmt_string = '{0:04d} {1: >6} {2: >6} {3: >6}'
    print('')
    print('{0: <4} {1: <6} {2: <6} {3: <6}'.format(' N', '   X', '   Y', '   Z'))
    for i in indices:
	X, Y, Z = Xs[i], Ys[i], Zs[i]
	print(fmt_string.format(i, f_(X), f_(Y), f_(Z)))


def fX(X, Y, Z, sigma, r, b):
    return -sigma * X + sigma * Y


def fY(X, Y, Z, sigma, r, b):
    return - X * Z + r * X - Y


def fZ(X, Y, Z, sigma, r, b):
    return X * Y - b * Z


def lorenz_solver(nt, dt, X0, Y0, Z0, sigma, r, b, scheme='da'):
    '''scheme:
    da: double approx (used by Lorenz
    ft: forward time
    rk4: Runge-Kutta 4th order'''
    if scheme not in ['da', 'ft', 'rk4']:
	raise Exception('scheme not recognized')

    Xs, Ys, Zs = [X0], [Y0], [Z0]
    X, Y, Z = X0, Y0, Z0
    times = np.linspace(0, nt * dt, nt)
    for t in times[1:]:
	if scheme == 'rk4':
            # RK4.
            kX1 = fX(X, Y, Z, sigma, r, b)
            kY1 = fY(X, Y, Z, sigma, r, b)
            kZ1 = fZ(X, Y, Z, sigma, r, b)

            kX2 = fX(X + dt/2 * kX1, Y + dt/2 * kY1, Z + dt/2 * kZ1, sigma, r, b)
            kY2 = fY(X + dt/2 * kX1, Y + dt/2 * kY1, Z + dt/2 * kZ1, sigma, r, b)
            kZ2 = fZ(X + dt/2 * kX1, Y + dt/2 * kY1, Z + dt/2 * kZ1, sigma, r, b)

            kX3 = fX(X + dt/2 * kX2, Y + dt/2 * kY1, Z + dt/2 * kZ1, sigma, r, b)
            kY3 = fY(X + dt/2 * kX2, Y + dt/2 * kY1, Z + dt/2 * kZ1, sigma, r, b)
            kZ3 = fZ(X + dt/2 * kX2, Y + dt/2 * kY1, Z + dt/2 * kZ1, sigma, r, b)

            kX4 = fX(X + dt * kX2, Y + dt * kY1, Z + dt * kZ1, sigma, r, b)
            kY4 = fY(X + dt * kX2, Y + dt * kY1, Z + dt * kZ1, sigma, r, b)
            kZ4 = fZ(X + dt * kX2, Y + dt * kY1, Z + dt * kZ1, sigma, r, b)

	    X = X + dt/6 * (kX1 + 2*kX2 + 2*kX3 + kX4)
	    Y = Y + dt/6 * (kY1 + 2*kY2 + 2*kY3 + kY4)
	    Z = Z + dt/6 * (kZ1 + 2*kZ2 + 2*kZ3 + kZ4)
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

    return times, np.array(Xs), np.array(Ys), np.array(Zs)

def plot_lorenz63_figures(ts, Xs, Ys, Zs):
    plt.figure(1)
    plt.clf()

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

    plt.figure(2)
    plt.clf()
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



def lorenz63_output(scheme='da'):
    '''Output similar tables/plots to Lorenz 63 paper.

    da: double approx (what Lorenz used)'''
    settings = dict(
	    X0=0,
	    Y0=1,
	    Z0=0,
	    sigma=10,
	    b=8/3,
	    r=28,
	    scheme=scheme)

    ts, Xs, Ys, Zs = lorenz_solver(6000, 0.01, **settings)

    print_equations()
    lgp_30_output(Xs, Ys, Zs)
    Z_extrema_indices = argrelextrema(Zs, np.greater)[0]
    lgp_30_output(Xs, Ys, Zs, indices=Z_extrema_indices)

    plot_lorenz63_figures(ts, Xs, Ys, Zs)


def chaos_demo(mode='IC', schemes=[]):
    '''Demonstrate chatoic behaviour when one of IC or scheme is changed'''
    settings = dict(
	    X0=0,
	    Y0=1,
	    Z0=0,
	    sigma=10,
	    b=8/3,
	    r=28,
	    scheme='da')

    if mode == 'scheme':
	settings['scheme'] = schemes[0]

    ts, Xs1, Ys1, Zs1 = lorenz_solver(6000, 0.01, **settings)
    if mode == 'IC':
	# slightly different ICs.
	settings['Y0'] = 1 + 1e-9
    elif mode == 'scheme':
	settings['scheme'] = schemes[1]
    ts, Xs2, Ys2, Zs2 = lorenz_solver(6000, 0.01, **settings)

    if mode == 'scheme':
	plt.figure('{}-{}'.format(mode, schemes))
    else:
	plt.figure(mode)
    plt.clf()
    plt.subplot(211)
    plt.title('Chaos when change in {}'.format(mode))
    plt.plot(ts, Ys1)
    plt.plot(ts, Ys2)
    plt.ylabel('Y')
    
    plt.subplot(212)
    plt.plot(ts, Ys2 - Ys1)
    plt.ylabel('$\Delta$ Y')
    plt.xlabel('time')


if __name__ == '__main__':
    lorenz63_output()

    chaos_demo(mode='IC')
    chaos_demo(mode='scheme', schemes=['da', 'rk4'])
    chaos_demo(mode='scheme', schemes=['da', 'ft'])
