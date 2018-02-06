# from sym import H, f, simple_trigs, t, Mext1, Mext3, w1, w2, w3, w4

import numpy as np

class Variables:
    phase = 0
    freq = 3.65

def boat_fish_model(state, t):
    x, y = state
    d_x = x * (2 - y - x)
    d_y = -y * (1 - 1.5 * x)
    return [d_x, d_y]


def pendulum_model(state, t):
    x, v = state
    d_x = v
    d_v = - x - 0.1 * v
    return [d_x, d_v]


def double_pendulum_model(state, t):
    from math import cos, sin
    m = 1
    l = 1
    g = 9.81
    t1, t2, p1, p2 = state
    d_t1 = 6 / m / l ** 2 * (2 * p1 - 3 * cos(t1 - t2) * p2) / (16 - 9 * cos(t1 - t2) ** 2)
    d_t2 = 6 / m / l ** 2 * (8 * p2 - 3 * cos(t1 - t2) * p1) / (16 - 9 * cos(t1 - t2) ** 2)
    d_p1 = -1 / 2 * m * l ** 2 * (d_t1 * d_t2 * sin(t1 - t2) + 3 * g / l * sin(t1))
    d_p2 = -1 / 2 * m * l ** 2 * (-d_t1 * d_t2 * sin(t1 - t2) + g / l * sin(t2))
    return [d_t1, d_t2, d_p1, d_p2]


def single_hand_model(state, t):

    print(t)

    # constant variables
    l1 = l2 = 0.5
    m1 = m2 = 5
    lc1 = l1 / 2
    lc2 = l2 / 2
    g = 9.81
    Ic1 = 1 / 3 * m1 * l1 ** 2
    Ic2 = 1 / 3 * m2 * l2 ** 2

    # load current state
    t1, t2, w1, w2, i1, i2 = state

    # replacement variables
    C1 = np.cos(t1)
    C2 = np.cos(t2)
    S1 = np.sin(t1)
    S2 = np.sin(t2)
    C12 = np.cos(t1 + t2)
    S12 = np.sin(t1 + t2)
    h12 = m2 * l1 * lc2 * S2
    G1 = m1 * g * lc1 * S1 + m2 * g * l1 * S1
    G12 = m2 * g * lc2 * S12
    H11 = Ic1 + m1 * lc1 ** 2 + Ic2 + m2 * (l1 ** 2 + l2 ** 2 + 2 * l1 * lc2 * C2)
    H22 = Ic2 + m2 * lc2 ** 2
    H12 = H21 = Ic2 + m2 * (lc2 ** 2 + l1 * lc2 * C2)
    H = np.array([[H11, H12], [H21, H22]])

    # external momenta
    wanted_t1 = 3
    wanted_t2 = 3
    # Mext1 = -30 * w1 - 50 * (t1 - wanted_t1) - 30 * i1
    # Mext2 = -10 * w2 - 20 * (t2 - wanted_t2) - 10 * i2
    Mext1 = 0
    Mext2 = 0
    # Mext1 = -5 * i1
    # Mext2 = -5 * i2

    HW_vector = [h12 * w2 ** 2 + 2 * h12 * w1 * w2 - G1 - G12 + Mext1,
                 -h12 * w1 ** 2 - G12 + Mext2]
    HW_vector = np.array(HW_vector)

    d_W = (np.linalg.inv(H).dot(HW_vector)).T

    d_t1 = w1
    d_t2 = w2
    d_w1 = d_W[0]
    d_w2 = d_W[1]

    return [d_t1, d_t2, d_w1, d_w2, t1 - wanted_t1, t2 - wanted_t2]


def heaviside(t):
    return 1 if t > 0 else 0


def impulse(t, duration):
    return heaviside(t) - heaviside(t - duration)


def double_hand_model(state, time):

    t1, t2, t3, t4, w1, w2, w3, w4, lambda_x, lambda_y = state
    theta = np.array([t1, t2, t3, t4])

    m1 = m2 = m3 = m4 = m = 5
    l1 = l2 = l3 = l4 = l = D = 0.5
    lc1 = lc2 = lc3 = lc4 = lc = l / 2
    g = 9.81
    nu = 100

    x1 = y1 = y3 = 0
    x3 = D

    J1 = J2 = J3 = J4 = J = 1 / 12 * m * l ** 2

    wanted1 = 3 * np.pi / 5
    wanted3 = np.pi - 3 * np.pi / 5
    phase = Variables.phase

    freq = Variables.freq # Hz
    omega = 2 * np.pi * freq
    ampl = 100

    # M1 = 0 - 10 * w1 + 1000 * (wanted1 - t1) + ampl * np.sin(omega * time) + 100
    # M3 = 0 - 10 * w3 - 1000 * (wanted3 - t3) + ampl * np.sin(omega * time + phase)
    # M1 = 0 - 10 * w1 + 1000 * (wanted1 - t1) + 100 * impulse(time, 0.1)
    # M3 = 0 - 10 * w3 - 1000 * (wanted3 - t3) - 100 * impulse(time, 0.1)
    # M1 = 100 * np.sin(40 * time)
    # M3 = 100 * np.sin(40 * time)
    M1 = 0
    M3 = 0

    C = np.cos(theta)
    C1, C2, C3, C4 = C
    S = np.sin(theta)
    S1, S2, S3, S4 = S
    C12 = np.cos(theta[0] + theta[1])
    C34 = np.cos(theta[2] + theta[3])
    S12 = np.sin(theta[0] + theta[1])
    S34 = np.sin(theta[2] + theta[3])

    H11 = J1 + m1 * lc1 ** 2 + J2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * C2)
    H22 = J2 + m2 * lc2 ** 2
    H12 = H21 = J2 + m2 * (lc2 ** 2 + l1 * lc2 * C2)
    h12 = m2 * l1 * lc2 * S2
    G1 = (m1 * lc1 ** 2 + m2 * l1) * g * C1
    G12 = m2 * lc2 * g * C12

    H33 = J3 + m3 * lc3 ** 2 + J4 + m4 * (l3 ** 2 + lc4 ** 2 + 2 * l3 * lc4 * C4)
    H44 = J4 + m4 * lc4 ** 2
    H34 = H43 = J4 + m4 * (lc4 ** 2 + l3 * lc4 * C4)
    h34 = m4 * l3 * lc4 * S[3]
    G3 = (m3 * lc3 ** 2 + m4 * l3) * g * C2
    G34 = m4 * lc4 * g * C34

    Jx1 = -l1 * S1 - l2 * S12
    Jy1 = l1 * C1 + l2 * C12
    Jx2 = - l2 * S12
    Jy2 = l2 * C12
    Jx3 = l3 * S3 + l4 * S34
    Jy3 = - l3 * C3 - l4 * C34
    Jx4 = l4 * S34
    Jy4 = - l4 * C34

    f1 = h12 * w2 ** 2 + 2 * h12 * w1 * w2 - G1 - G12 + M1
    f2 = -h12 * w1 ** 2 - G12
    f3 = h34 * w3 ** 2 + 2 * h34 * w3 * w4 - G3 - G34 + M3
    f4 = -h34 * w3 ** 2 - G34

    X = l1 * C1 + l2 * C12 - l3 * C3 - l4 * C34 + x1 - x3
    Y = l1 * S1 + l2 * S12 - l3 * S3 - l4 * S34 + y1 - y3

    gx = - (Jy1 * w1 ** 2 + Jy2 * w2 ** 2 + 2 * Jy2 * w1 * w2) \
         + (Jy3 * w3 ** 2 + Jy4 * w4 ** 2 + 2 * Jy4 * w3 * w4) \
         + 2 * nu * (Jx1 * w1 + Jx2 * w2 - Jx3 * w3 - Jx4 * w4) \
         + nu ** 2 * X

    gy = + (Jx1 * w1 ** 2 + Jx2 * w2 ** 2 + 2 * Jx2 * w1 * w2) \
         - (Jx3 * w3 ** 2 + Jx4 * w4 ** 2 + 2 * Jx4 * w3 * w4) \
         + 2 * nu * (Jy1 * w1 + Jy2 * w2 - Jy3 * w3 - Jy4 * w4) \
         + nu ** 2 * Y

    # M * d_state = b
    M = np.array([[  1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ],
                  [  0 ,  1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ],
                  [  0 ,  0 , -1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ],
                  [  0 ,  0 ,  0 , -1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ],
                  [  0 ,  0 ,  0 ,  0 , H11, H12,  0 ,  0 ,-Jx1,-Jy1],
                  [  0 ,  0 ,  0 ,  0 , H21, H22,  0 ,  0 ,-Jx2,-Jy2],
                  [  0 ,  0 ,  0 ,  0 ,  0 ,  0 , H33, H34, Jx3, Jy3],
                  [  0 ,  0 ,  0 ,  0 ,  0 ,  0 , H43, H44, Jx4, Jy4],
                  [  0 ,  0 ,  0 ,  0 ,-Jx1,-Jx2, Jx3, Jx4,  0 ,  0 ],
                  [  0 ,  0 ,  0 ,  0 ,-Jy1,-Jy2, Jy3, Jy4,  0 ,  0 ]])

    b = np.array([[w1], [w2], [w3], [w4], [f1], [f2], [f3], [f4], [gx], [gy]])

    d_state = np.linalg.inv(M).dot(b)

    #print(d_state.T[0].tolist())

    return d_state.T[0].tolist()


# def double_hand_model(state, time):
#     import math
#
#     print(time)
#
#     state_subs = [st[1] for st in simple_trigs]
#     state_subs = [(x, 0) for x in state_subs]
#
#     state_subs[0] = (state_subs[0][0], math.cos(state[6 + 0]))
#     state_subs[1] = (state_subs[1][0], math.cos(state[6 + 1]))
#     state_subs[2] = (state_subs[2][0], math.cos(state[6 + 2]))
#     state_subs[3] = (state_subs[3][0], math.cos(state[6 + 3]))
#
#     state_subs[4] = (state_subs[4][0], math.sin(state[6 + 0]))
#     state_subs[5] = (state_subs[5][0], math.sin(state[6 + 1]))
#     state_subs[6] = (state_subs[6][0], math.sin(state[6 + 2]))
#     state_subs[7] = (state_subs[7][0], math.sin(state[6 + 3]))
#
#     state_subs[8] = (state_subs[8][0], math.cos(state[6 + 0] + state[6 + 1]))
#     state_subs[9] = (state_subs[9][0], math.sin(state[6 + 0] + state[6 + 1]))
#     state_subs[10] = (state_subs[10][0], math.cos(state[6 + 2] + state[6 + 3]))
#     state_subs[11] = (state_subs[11][0], math.sin(state[6 + 2] + state[6 + 3]))
#
#     w_subs = [(w1, state[0]),
#               (w2, state[1]),
#               (w3, state[2]),
#               (w4, state[3])]
#
#     M_subs = [(Mext1, 0),
#               (Mext3, 0)]
#
#     # TODO: lambdify H & f
#
#     dX = H.subs(state_subs).inv() * f.subs(state_subs + w_subs + M_subs)
#
#     dX_list = [dX[i, 0] for i in range(10)]
#
#     return dX_list


curr_model = boat_fish_model

# H = pickle.load(open('H.dat', 'rb'))
# f = pickle.load(open('f.dat', 'rb'))
# constsubs = pickle.load(open('constsubs.dat', 'rb'))
# simple_trigs = pickle.load(open('simple_trigs.dat', 'rb'))
