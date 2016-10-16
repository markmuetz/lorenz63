import re
import numpy as np
import pylab as plt
from sympy import symbols, python, Eq, Derivative, init_printing, solve, lambdify

def build_equations():
    X, Y, Z, t = symbols('X Y Z t')
    sigma, r, b = symbols('sigma r b')
    eq1 = Eq(Derivative(X, t), -sigma * X + sigma * Y)
    eq2 = Eq(Derivative(Y, t), -X*Z + r*X -Y)
    eq3 = Eq(Derivative(Z, t), X*Y - b*Z)

    return X, Y, Z, t, sigma, r, b, eq1, eq2, eq3


def print_equations(eq1, eq2, eq3):
    init_printing()
    print(eq1)
    print(eq2)
    print(eq3)


def build_ft_soln(X, Y, Z, t, sigma, r, b, eq1, eq2, eq3):
    Xn, Yn, Zn, dt = symbols('Xn Yn Zn dt')
    # discrete eqn:
    deq1 = Eq((Xn - X)/dt, eq1.rhs)
    deq2 = Eq((Yn - Y)/dt, eq2.rhs)
    deq3 = Eq((Zn - Z)/dt, eq3.rhs)
    s1 = solve(deq1, Xn)[0]
    s2 = solve(deq2, Yn)[0]
    s3 = solve(deq3, Zn)[0]
    return dt, s1, s2, s3


def gen_python(s1, s2, s3, X, Y, Z, sigma, r, b, dt, nt):
    py_tpl = '''
import numpy as np

X = {X0}
Y = {Y0}
Z = {Z0}
sigma = {sigma}
r = {r}
b = {b}
dt = {dt}
nt = {nt}
ts = np.linspace(0, nt*dt, nt + 1)
soln = [(0, X, Y, Z)]
for t in ts[1:]:
    print(t)
    {s1}
    {s2}
    {s3}
    X, Y, Z = Xn, Yn, Zn
    soln.append((t, X, Y, Z))
soln = np.array(soln)
'''
    # Values:
    Xv, Yv, Zv = 0, 1, 0
    sigmav, bv, rv = 10, 8./3, 28
    dtv = 0.01
    s1p = re.sub('^e', 'Xn', python(s1).split('\n')[-1])
    s2p = re.sub('^e', 'Yn', python(s2).split('\n')[-1])
    s3p = re.sub('^e', 'Zn', python(s3).split('\n')[-1])
    py_str = py_tpl.format(X0=Xv, Y0=Yv, Z0=Zv, sigma=sigmav, r=rv, b=bv, dt=dtv, nt=nt,
                           s1=s1p, s2=s2p, s3=s3p)
    exec(compile(py_str, '<string>', 'exec'))
    return soln

    
def solve_eqns(s1, s2, s3, X, Y, Z, sigma, r, b, dt, nt):
    # Values:
    Xv, Yv, Zv = 0, 1, 0
    fX = lambdify((X, Y, Z, sigma, r, b, dt), s1)
    fY = lambdify((X, Y, Z, sigma, r, b, dt), s2)
    fZ = lambdify((X, Y, Z, sigma, r, b, dt), s3)
    sigmav, bv, rv = 10, 8./3, 28
    soln = [(0, Xv, Yv, Zv)]
    dtv = 0.01
    dts = np.linspace(0, nt*dtv, nt + 1)
    for t in dts[1:]:
        print(t)
        # WAAY faster.
        Xnv = fX(Xv, Yv, Zv, sigmav, rv, bv, dtv)
        Ynv = fY(Xv, Yv, Zv, sigmav, rv, bv, dtv)
        Znv = fZ(Xv, Yv, Zv, sigmav, rv, bv, dtv)
        #Xnv = s1.subs([(X, Xv), (Y, Yv), (Z, Zv), (sigma, sigmav), (r, rv), (b, bv), (dt, dtv)])
        #Ynv = s2.subs([(X, Xv), (Y, Yv), (Z, Zv), (sigma, sigmav), (r, rv), (b, bv), (dt, dtv)])
        #Znv = s3.subs([(X, Xv), (Y, Yv), (Z, Zv), (sigma, sigmav), (r, rv), (b, bv), (dt, dtv)])
        Xv, Yv, Zv = Xnv, Ynv, Znv
        soln.append((t, Xv, Yv, Zv))
    return np.array(soln)


if __name__ == '__main__':
    X, Y, Z, t, sigma, r, b, eq1, eq2, eq3 = build_equations()
    print_equations(eq1, eq2, eq3)
    dt, s1, s2, s3 = build_ft_soln(X, Y, Z, t, sigma, r, b, eq1, eq2, eq3)
    nt = 6000
    # Solves equations using sympy.lambdify.
    soln = solve_eqns(s1, s2, s3, X, Y, Z, sigma, r, b, dt, nt=nt)
    # Generates then executes python so solve equations.
    soln2 = gen_python(s1, s2, s3, X, Y, Z, sigma, r, b, dt, nt=nt)

    plt.figure(1)
    plt.plot(soln[:, 0], soln[:, 2])
    plt.plot(soln2[:, 0], soln2[:, 2])
    plt.figure(2)
    plt.plot(soln2[:, 0], soln[:, 2] - soln2[:, 2])
    plt.show()
