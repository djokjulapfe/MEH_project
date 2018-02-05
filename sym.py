import sympy
import pickle
import math

sympy.init_printing()


def remove_elements(eq, list):
    ret = eq
    for var in list:
        ret = ret - eq.coeff(var) * var
    return sympy.expand(ret)


def print_LaTeX(name, eq):
    # print('\\begin{equation}')
    # print(name, '=')
    # print(sympy.latex(eq.subs(simple_trigs)))
    # print('\\end{equation}')
    pass


# Helper constants
c_half = sympy.Rational(1, 2)
c_third = sympy.Rational(1, 3)
c_twelfth = sympy.Rational(1, 12)

# time symbol
t = sympy.symbols('t')

# symbols for constants
l1, l2, l3, l4, D = sympy.symbols('l_1 l_2 l_3 l_4 D')
m1, m2, m3, m4 = sympy.symbols('m_1 m_2 m_3 m_4')
g = sympy.symbols('g')
nu = sympy.symbols('nu')

# Momenta of inertia
Ic1 = c_twelfth * m1 * l1 ** 2
Ic2 = c_twelfth * m2 * l2 ** 2
Ic3 = c_twelfth * m3 * l3 ** 2
Ic4 = c_twelfth * m4 * l4 ** 2

# Angles as functions of time
t1 = sympy.Function('theta_1')(t)
t2 = sympy.Function('theta_2')(t)
t3 = sympy.Function('theta_3')(t)
t4 = sympy.Function('theta_4')(t)

# External Momenta
Mext1 = sympy.Function('M_1')(t)
Mext3 = sympy.Function('M_3')(t)

# Derivatives of angles
dt1 = sympy.diff(t1)
dt2 = sympy.diff(t2)
dt3 = sympy.diff(t3)
dt4 = sympy.diff(t4)

# Helper symbols for trigonometric functions
C1 = sympy.cos(t1)
C2 = sympy.cos(t2)
C3 = sympy.cos(t3)
C4 = sympy.cos(t4)

S1 = sympy.sin(t1)
S2 = sympy.sin(t2)
S3 = sympy.sin(t3)
S4 = sympy.sin(t4)

C12 = sympy.cos(t1 + t2)
C34 = sympy.cos(t3 + t4)
S12 = sympy.sin(t1 + t2)
S34 = sympy.sin(t3 + t4)

simple_trigs = [(sympy.cos(t1), sympy.Symbol('C1')),
                (sympy.cos(t2), sympy.Symbol('C2')),
                (sympy.cos(t3), sympy.Symbol('C3')),
                (sympy.cos(t4), sympy.Symbol('C4')),

                (sympy.sin(t1), sympy.Symbol('S1')),
                (sympy.sin(t2), sympy.Symbol('S2')),
                (sympy.sin(t3), sympy.Symbol('S3')),
                (sympy.sin(t4), sympy.Symbol('S4')),

                (sympy.cos(t1 + t2), sympy.Symbol('C12')),
                (sympy.sin(t1 + t2), sympy.Symbol('S12')),
                (sympy.cos(t3 + t4), sympy.Symbol('C34')),
                (sympy.sin(t3 + t4), sympy.Symbol('S34')), ]

# print('Created symbols')

# Positions of hands' centre of mass
x1 = sympy.Matrix([0, 0])
x3 = sympy.Matrix([D, 0])
xc1 = x1 + l1 / 2 * sympy.Matrix([C1, S1])
xc2 = x1 + l1 * sympy.Matrix([C1, S1]) + l2 / 2 * sympy.Matrix([C12, S12])
xc3 = x3 + l3 / 2 * sympy.Matrix([C3, C4])
xc4 = x3 + l3 * sympy.Matrix([C3, S4]) + l4 / 2 * sympy.Matrix([C34, S34])

print_LaTeX('x_1', x1)
print_LaTeX('x_3', x3)
print_LaTeX('x_{c1}', xc1)
print_LaTeX('x_{c2}', xc2)
print_LaTeX('x_{c3}', xc3)
print_LaTeX('x_{c4}', xc4)

# print('Created coordinate symbols')

# derivatives of the positions
dxc1 = sympy.diff(xc1, t)
dxc2 = sympy.diff(xc2, t)
dxc3 = sympy.diff(xc3, t)
dxc4 = sympy.diff(xc4, t)

print_LaTeX('dx_{c1}', dxc1)
print_LaTeX('dx_{c2}', dxc2)
print_LaTeX('dx_{c3}', dxc3)
print_LaTeX('dx_{c4}', dxc4)

# print('Calculated derivatives of coordinates')

# Kinetic energy of each hand
Tc1 = sympy.simplify(c_half * m1 * dxc1.T * dxc1 + sympy.Matrix([c_half * Ic1 * dt1 ** 2]))
Tc2 = sympy.simplify(c_half * m2 * dxc2.T * dxc2 + sympy.Matrix([c_half * Ic2 * (dt1 + dt2) ** 2]))
Tc3 = sympy.simplify(c_half * m3 * dxc3.T * dxc3 + sympy.Matrix([c_half * Ic3 * dt3 ** 2]))
Tc4 = sympy.simplify(c_half * m4 * dxc4.T * dxc4 + sympy.Matrix([c_half * Ic4 * (dt3 + dt4) ** 2]))

print_LaTeX('T_{c1}', Tc1)
print_LaTeX('T_{c2}', Tc2)
print_LaTeX('T_{c3}', Tc3)
print_LaTeX('T_{c4}', Tc4)

# Total kinetic energy
T = Tc1 + Tc2 + Tc3 + Tc4
T = sympy.expand(T[0])

print_LaTeX('T', T)

# print('Kinetic energy calculated')

# Potential energy of each hand
# TODO: put xc1[1] instead of l1/2*S1
U1 = l1 / 2 * S1 * m1 * g
U2 = m2 * g * (l1 * S1 + l2 / 2 * S12)
U3 = l3 / 2 * S3 * m3 * g
U4 = m4 * g * (l3 * S3 + l4 / 2 * S34)

print_LaTeX('U_1', U1)
print_LaTeX('U_2', U2)
print_LaTeX('U_3', U3)
print_LaTeX('U_4', U4)

# Total potential energy
U = U1 + U2 + U3 + U4
U = sympy.expand(U)

print_LaTeX('U', U)

# print("Potential energy calculated")

# Work of the external momenta
W = Mext1 * t1 + Mext3 * t3

print_LaTeX('W', W)

# Constrictions
X = x1[0] + l1 * C1 + l2 * C12 - x3[0] - l3 * C3 - l4 * C34
Y = x1[1] + l1 * S1 + l2 * S12 - x3[1] - l3 * S3 - l4 * S34
lambdaX = sympy.symbols('lambda_x')
lambdaY = sympy.symbols('lambda_y')

print_LaTeX('X', X)
print_LaTeX('Y', Y)

# print('Constraints created')

# Lagrangian of the system
L = T - U + W + lambdaX * X + lambdaY * Y
L = sympy.expand(L)

print_LaTeX('L', L)

# print('Lagrangian created')

w1 = sympy.Function('w_1')(t)
w2 = sympy.Function('w_2')(t)
w3 = sympy.Function('w_3')(t)
w4 = sympy.Function('w_4')(t)
dtsub = [(dt1, w1), (dt2, w2), (dt3, w3), (dt4, w4)]

# Equations of motion
# TODO: create a list instead of individual equations
Eq1 = (L.diff(t1) - L.diff(dt1).diff(t)).subs(simple_trigs)
Eq2 = (L.diff(t2) - L.diff(dt2).diff(t)).subs(simple_trigs)
Eq3 = (L.diff(t3) - L.diff(dt3).diff(t)).subs(simple_trigs)
Eq4 = (L.diff(t4) - L.diff(dt4).diff(t)).subs(simple_trigs)
Eq5 = (X.diff(t).diff(t) + 2 * nu * X.diff(t) + nu ** 2 * X).subs(simple_trigs)
Eq6 = (Y.diff(t).diff(t) + 2 * nu * Y.diff(t) + nu ** 2 * Y).subs(simple_trigs)
Eq7 = (dt1 - w1).subs(simple_trigs)
Eq8 = (dt2 - w2).subs(simple_trigs)
Eq9 = (dt3 - w3).subs(simple_trigs)
Eq10 = (dt4 - w4).subs(simple_trigs)

H = sympy.eye(10, 10)

# Inertial matrix
H[0, 0] = -Eq1.coeff(dt1.diff(t))
H[0, 1] = H[1, 0] = -Eq1.coeff(dt2.diff(t))
H[1, 1] = -Eq2.coeff(dt2.diff(t))
h12 = Eq1.coeff(dt2**2)

H[2, 2] = -Eq3.coeff(dt3.diff(t))
H[2, 3] = H[3, 2] = -Eq3.coeff(dt4.diff(t))
H[3, 3] = -Eq4.coeff(dt4.diff(t))
h34 = Eq3.coeff(dt4**2)

# Jacobian matrix
H[0, 4] = H[4, 0] = -Eq1.coeff(lambdaX)
H[0, 5] = H[5, 0] = -Eq1.coeff(lambdaY)
H[1, 4] = H[4, 1] = -Eq2.coeff(lambdaX)
H[1, 5] = H[5, 1] = -Eq2.coeff(lambdaY)

H[2, 4] = H[4, 2] = Eq3.coeff(lambdaX)
H[2, 5] = H[5, 2] = Eq3.coeff(lambdaY)
H[3, 4] = H[4, 3] = Eq4.coeff(lambdaX)
H[3, 5] = H[5, 3] = Eq4.coeff(lambdaY)

H[4, 4] = H[5, 5] = 0

# RHS
f = sympy.zeros(10, 1)

vars = [lambdaX, lambdaY, dt1.diff(t), dt2.diff(t), dt3.diff(t), dt4.diff(t)]
f[0] = remove_elements(sympy.expand(Eq1), vars).subs(dtsub)
f[1] = remove_elements(sympy.expand(Eq2), vars).subs(dtsub)
f[2] = remove_elements(sympy.expand(Eq3), vars).subs(dtsub)
f[3] = remove_elements(sympy.expand(Eq4), vars).subs(dtsub)
f[4] = remove_elements(sympy.expand(Eq5), vars).subs(dtsub)
f[5] = remove_elements(sympy.expand(Eq6), vars).subs(dtsub)
f[6] = w1
f[7] = w2
f[8] = w3
f[9] = w4

constsubs = [(l1, sympy.Rational(1, 2)), (l2, sympy.Rational(1, 2)),
             (l3, sympy.Rational(1, 2)), (l4, sympy.Rational(1, 2)),
             (D, sympy.Rational(1, 2)),
             (m1, 5), (m2, 5), (m3, 5), (m4, 5),
             (g, 5), (nu, 1000)]

H = H.subs(constsubs)
f = f.subs(constsubs)

# example_state = [3 * math.pi / 5, - math.pi + 3 * math.pi / 5, math.pi - 3 * math.pi / 5, math.pi - 3 * math.pi / 5]
# example_state_subs = [st[1] for st in simple_trigs]
# example_state_subs = [(x, 0) for x in example_state_subs]
#
# example_state_subs[0] = (example_state_subs[0][0], math.cos(example_state[0]))
# example_state_subs[1] = (example_state_subs[1][0], math.cos(example_state[1]))
# example_state_subs[2] = (example_state_subs[2][0], math.cos(example_state[2]))
# example_state_subs[3] = (example_state_subs[3][0], math.cos(example_state[3]))
#
# example_state_subs[4] = (example_state_subs[4][0], math.sin(example_state[0]))
# example_state_subs[5] = (example_state_subs[5][0], math.sin(example_state[1]))
# example_state_subs[6] = (example_state_subs[6][0], math.sin(example_state[2]))
# example_state_subs[7] = (example_state_subs[7][0], math.sin(example_state[3]))
#
# example_state_subs[8] = (example_state_subs[8][0], math.cos(example_state[0] + example_state[1]))
# example_state_subs[9] = (example_state_subs[9][0], math.sin(example_state[0] + example_state[1]))
# example_state_subs[10] = (example_state_subs[10][0], math.cos(example_state[2] + example_state[3]))
# example_state_subs[11] = (example_state_subs[11][0], math.sin(example_state[2] + example_state[3]))

# pickle.dump(H, open('H.dat', 'wb'))
# pickle.dump(f, open('f.dat', 'wb'))
# pickle.dump(constsubs, open('constsubs.dat', 'wb'))
# pickle.dump(simple_trigs, open('simple_trigs.dat', 'wb'))



'''
Hinv = pickle.load(open('M6.dat', 'rb'))

Hinv_subs = [(sympy.Symbol('M{}{}'.format(i, j)), H[i, j]) for i in range(6) for j in range(6)]

print_LaTeX('Eq1', Eq1)
print_LaTeX('Eq2', Eq2)
print_LaTeX('Eq3', Eq3)
print_LaTeX('Eq4', Eq4)
print_LaTeX('Eq5', Eq5)
print_LaTeX('Eq6', Eq6)
print_LaTeX('Eq7', Eq7)
print_LaTeX('Eq8', Eq8)
print_LaTeX('Eq9', Eq9)
print_LaTeX('Eq10', Eq10)

# print("Equations of motion created")

# Substitutions

# Eq1 = Eq1.subs(constsubs)
# Eq2 = Eq2.subs(constsubs)
# Eq3 = Eq3.subs(constsubs)
# Eq4 = Eq4.subs(constsubs)
# Eq5 = Eq5.subs(constsubs)
# Eq6 = Eq6.subs(constsubs)
# Eq7 = Eq7.subs(constsubs)
# Eq8 = Eq8.subs(constsubs)
# Eq9 = Eq9.subs(constsubs)
# Eq10 = Eq10.subs(constsubs)

# print("Substituting constants")

# Extract variables
# solutions = sympy.solve((Eq1, Eq2, Eq3, Eq4, Eq5, Eq6, Eq7, Eq8, Eq9, Eq10),
#                         dt1, dt2, dt3, dt4,
#                         w1.diff(t), w2.diff(t), w3.diff(t), w4.diff(t),
#                         lambdaX, lambdaY)
# solutions = sympy.solve((Eq1, Eq2, Eq3, Eq4, Eq5, Eq6),
#                         dt1, dt2, dt3, dt4, lambdaX, lambdaY)

# print("System of equations solved")


# sympy.pprint(solutions)
'''
