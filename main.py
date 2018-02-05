import numpy as np
from scipy.integrate import odeint
import models
from matplotlib import pyplot as plt
from matplotlib import animation
import math

t = np.arange(0, 1, 0.001)
angPentagon = 3 * math.pi / 5
init_state = [angPentagon, - math.pi + angPentagon, math.pi - angPentagon, math.pi - angPentagon, 0, 0, 0, 0, 0, 0]
#init_state = [2, 1]

models.curr_model = models.double_hand_model
state = odeint(models.curr_model, init_state, t)

# ANIMATION:
# state = np.array([[sin(x) + sin(y), -cos(x) - cos(y)] for x, y in state[:, :2]])
# state = np.array([[np.sin(x), -np.cos(x), np.sin(x) + np.sin(y), -np.cos(x) - np.cos(y)] for x, y in state[:, :2]])
# state = np.array([[x, 1, y, -1] for x, y in state[:, :2]])

def animate_system(state):

    l = D = 0.5
    state = np.array([[[0, 0],
                       [l * math.cos(t1), l * math.sin(t1)],
                       [l * math.cos(t1) + l * math.cos(t1 + t2), l * math.sin(t1) + l * math.sin(t1 + t2)],
                       [D + l * math.cos(t3) + l * math.cos(t3 + t4), l * math.sin(t3) + l * math.sin(t3 + t4)],
                       [D + l * math.cos(t3), l * math.sin(t3)],
                       [D, 0]] for t1, t2, t3, t4 in state[:, :4]])

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)


    def init():
        """initialize animation"""
        line.set_data([], [])
        return line,


    def animate(i):
        """perform animation step"""
        data = [[0, 0]]
        for vec in state[i]:
            data += [vec]
        data = np.array(data).T
        # line.set_data([[0, state[i][0]], [0, state[i][1]]])
        line.set_data(data)
        return line,


    ani = animation.FuncAnimation(fig, animate, frames=len(state), interval=1000 / 60, blit=True, init_func=init)

    # ani.save("anim.mp4", fps = 60, extra_args=['-vcodec', 'libx264'])

    plt.show()

animate_system(state)