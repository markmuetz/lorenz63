from sympy import *

def build_equations():
    X, Y, Z, t = symbols('X Y Z t')
    sigma, r, b = symbols('sigma r b')
    eq1 = Eq(Derivative(X, t), -sigma * X + sigma * Y)
    eq2 = Eq(Derivative(Y, t), -X*Z + r*X -Y)
    eq3 = Eq(Derivative(Z, t), X*Y - b*Z)

    return X, Y, Z, t, sigma, r, b, eq1, eq2, eq3

def print_equations():
    init_printing()
    X, Y, Z, t, sigma, r, b, eq1, eq2, eq3 = build_equations()
    print(eq1)
    print(eq2)
    print(eq3)

if __name__ == '__main__':
    print_equations()
