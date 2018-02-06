import math

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.integrate import odeint

import models

t = np.arange(0, 10, 0.001)
angPentagon = 3 * math.pi / 5
regular_pentagon_state = [angPentagon, - math.pi + angPentagon, math.pi - angPentagon, math.pi - angPentagon, 0, 0, 0, 0, 0, 0]
inverted_pentagon_state = [-angPentagon, math.pi - angPentagon, -math.pi + angPentagon, - math.pi + angPentagon, 0, 0, 0, 0, 0, 0]
stable_state = [0.5, -1, -0.5, 1, 0, 0, 0, 0, 0, 0]


def get_freq(signal, freq, sample_rate):
    real, imag = 0, 0
    dt = 1.0 / sample_rate

    for i in range(len(signal)):
        t = i / sample_rate
        real += signal[i] * np.cos(-2 * np.pi * t * freq) * dt
        imag += signal[i] * np.sin(-2 * np.pi * t * freq) * dt

    return np.sqrt(real ** 2 + imag ** 2) / len(signal) * sample_rate


def simulate():
    init_state = regular_pentagon_state
    # init_state = [2, 1]

    models.curr_model = models.double_hand_model
    return odeint(models.curr_model, init_state, t)


# ANIMATION:
# state = np.array([[sin(x) + sin(y), -cos(x) - cos(y)] for x, y in state[:, :2]])
# state = np.array([[np.sin(x), -np.cos(x), np.sin(x) + np.sin(y), -np.cos(x) - np.cos(y)] for x, y in state[:, :2]])
# state = np.array([[x, 1, y, -1] for x, y in state[:, :2]])

def animate_system(state):
    l = D = 0.5
    # state = [[[0, 0],
    #           [l * math.cos(t1), l * math.sin(t1)],
    #           [l * math.cos(t1) + l * math.cos(t1 + t2), l * math.sin(t1) + l * math.sin(t1 + t2)],
    #           [D + l * math.cos(t3) + l * math.cos(t3 + t4), l * math.sin(t3) + l * math.sin(t3 + t4)],
    #           [D + l * math.cos(t3), l * math.sin(t3)],
    #           [D, 0]] for t1, t2, t3, t4 in state[:, :4]]

    wanted_state = [[[0, 0],
                     [l * math.cos(t1), l * math.sin(t1)],
                     [l * math.cos(t1) + l * math.cos(t1 + t2), l * math.sin(t1) + l * math.sin(t1 + t2)],
                     [D + l * math.cos(t3) + l * math.cos(t3 + t4), l * math.sin(t3) + l * math.sin(t3 + t4)],
                     [D + l * math.cos(t3), l * math.sin(t3)],
                     [D, 0]] for t1, t2, t3, t4 in [regular_pentagon_state[:4]]]

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-0.5, 1), ylim=(-1, 1))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    wanted_line, = ax.plot([], [], 'o-', lw=2)

    def init():
        """initialize animation"""
        line.set_data([], [])
        wanted_line.set_data([], [])
        return line, wanted_line

    def animate(i):
        """perform animation step"""

        data = [[0, 0]]
        for vec in state[i]:
            data += [vec]
        data = np.array(data).T
        line.set_data(data)

        data = [[0, 0]]
        for vec in wanted_state[0]:
            data += [vec]
        data = np.array(data).T
        wanted_line.set_data(data)

        return wanted_line, line

    ani = animation.FuncAnimation(fig, animate, frames=len(state), interval=1000 / 60, blit=True, init_func=init)

    # ani.save("anim.mp4", fps = 60, extra_args=['-vcodec', 'libx264'])

    plt.show()


def plot5():

    simulation_state = simulate()

    l = D = 0.5
    positions = [[[0, 0],
                  [l * math.cos(t1), l * math.sin(t1)],
                  [l * math.cos(t1) + l * math.cos(t1 + t2), l * math.sin(t1) + l * math.sin(t1 + t2)],
                  [D + l * math.cos(t3) + l * math.cos(t3 + t4), l * math.sin(t3) + l * math.sin(t3 + t4)],
                  [D + l * math.cos(t3), l * math.sin(t3)],
                  [D, 0]] for t1, t2, t3, t4 in simulation_state[:, :4]]

    five = np.array(positions)[:, 3]

    plt.xlim(-0.5, 1)
    plt.ylim(-1, 1)
    plt.plot(five[:, 0], five[:, 1], label = 'joint #5')
    plt.savefig('plotXY.png')
    plt.clf()
    # plt.show()


def plot_x_and_y():

    simulation_state = simulate()

    l = D = 0.5
    positions = [[[0, 0],
                  [l * math.cos(t1), l * math.sin(t1)],
                  [l * math.cos(t1) + l * math.cos(t1 + t2), l * math.sin(t1) + l * math.sin(t1 + t2)],
                  [D + l * math.cos(t3) + l * math.cos(t3 + t4), l * math.sin(t3) + l * math.sin(t3 + t4)],
                  [D + l * math.cos(t3), l * math.sin(t3)],
                  [D, 0]] for t1, t2, t3, t4 in simulation_state[:, :4]]

    five = np.array(positions)[:, 3]

    plt.ylim(-1, 1)
    p1, = plt.plot(t, five[:, 0], label = 'x')
    p2, = plt.plot(t, five[:, 1], label = 'y')
    plt.legend(handles = [p1, p2])
    plt.xlabel('t[s]')
    plt.ylabel('a[m]')
    plt.grid()
    plt.xlim(0, t.max())
    plt.savefig('plot(t).png')
    # plt.show()


def play_animation():

    simulation_state = simulate()

    l = D = 0.5
    positions = [[[0, 0],
                  [l * math.cos(t1), l * math.sin(t1)],
                  [l * math.cos(t1) + l * math.cos(t1 + t2), l * math.sin(t1) + l * math.sin(t1 + t2)],
                  [D + l * math.cos(t3) + l * math.cos(t3 + t4), l * math.sin(t3) + l * math.sin(t3 + t4)],
                  [D + l * math.cos(t3), l * math.sin(t3)],
                  [D, 0]] for t1, t2, t3, t4 in simulation_state[:, :4]]

    animate_system(positions)


def saveRange(N):

    for i in range(N):

        if i % 10 == 0:
            print(100 * i / N)


        def dist(x):
            return 4 * (x - 0.5) ** 3 + 0.5


        models.Variables.phase = np.pi * dist(i / N)

        simulation_state = simulate()

        l = D = 0.5
        positions = [[[0, 0],
                      [l * math.cos(t1), l * math.sin(t1)],
                      [l * math.cos(t1) + l * math.cos(t1 + t2), l * math.sin(t1) + l * math.sin(t1 + t2)],
                      [D + l * math.cos(t3) + l * math.cos(t3 + t4), l * math.sin(t3) + l * math.sin(t3 + t4)],
                      [D + l * math.cos(t3), l * math.sin(t3)],
                      [D, 0]] for t1, t2, t3, t4 in simulation_state[:, :4]]

        five = np.array(positions)[:, 3]

        plt.xlim(0, 0.5)
        plt.ylim(0, 1)
        plt.plot(five[:, 0], five[:, 1])
        plt.savefig('phase{}.png'.format(str(i).zfill(3)))
        plt.clf()
        plt.show()


def plot_angles():

    init_state = np.array(inverted_pentagon_state)

    simulation_state = simulate()
    simulation_state = np.array([ss - init_state for ss in simulation_state])

    # for f in np.arange(3.5, 3.71, 0.01):
    #
    #     models.Variables.freq = f
    #
    #     simulation_state = simulate()
    #     simulation_state = np.array([ss - init_state for ss in simulation_state])
    #
    #     print(f, get_freq(simulation_state[:, 0], models.Variables.freq, 100))


    p1, = plt.plot(t, simulation_state[:, 0], label='theta 1')
    p2, = plt.plot(t, simulation_state[:, 1], label='theta 2')
    p3, = plt.plot(t, simulation_state[:, 2], label='theta 3')
    p4, = plt.plot(t, simulation_state[:, 3], label='theta 4')
    plt.legend(handles = [p1, p2, p3, p4])
    plt.savefig('example_angles.png')
    plt.show()


def freqsweep(start, end, logarithmic=True):
    freq = np.logspace(np.log10(start), np.log10(end), num=100)

    vals = []

    for f in freq:

        print(f)

        models.Variables.freq = f
        simulation_state = simulate()

        vals.append(get_freq(simulation_state[5000:, 0], f, 1000))

    plt.semilogx(freq, vals)
    plt.xlabel('freq[Hz]')
    plt.ylabel('A[rad]')
    plt.savefig('freqsweep.png')
    plt.show()


models.Variables.freq = 4
models.Variables.phase = np.pi

# plot_angles()
# models.Variables.freq = 3
# play_animation()
# plot5()
# plot_x_and_y()
freqsweep(2, 10)