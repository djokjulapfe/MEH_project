import sympy
import pickle

sympy.init_printing()


def getMatrix(n):
    M = sympy.zeros(n, n)

    for i in range(n):
        for j in range(n):
            M[i, j] = sympy.Symbol('M{}{}'.format(i, j))

    return M


M = getMatrix(6)
M[2, 0] = M[2, 1] = M[3, 0] = M[3, 1] = M[0, 2] = M[0, 3] = M[1, 2] = M[1, 3] = M[4, 4] = M[4, 5] = M[5, 4] = M[
    5, 5] = 0

Minv = M.inv()
pickle.dump(Minv, open('M6.dat', 'wb'))