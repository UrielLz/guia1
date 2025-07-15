import numpy as np

def funciones(x, problema_id):
    if problema_id == 0:
        # Problema 0
        f1 = 4 * (x[0] ** 2) + 4 * (x[1] ** 2)
        f2 = (x[0] - 5) ** 2 + (x[1] - 5) ** 2
        return f1, f2

    elif problema_id == 1:
        # Problema 1
        f1 = x[0] ** 2
        f2 = (x[0] - 2) ** 2
        return f1, f2

    elif problema_id == 2:
        # Problema 2
        f1 = x[0] ** 2 + x[1]
        f2 = x[1] ** 2 + x[0]
        return f1, f2

    elif problema_id == 3:
        # Problema 3 (Maximización transformada)
        f1 = -(0.4 * x[0] + 0.3 * x[1])
        f2 = -x[0]
        return f1, f2

    else:
        raise ValueError("ID de problema no reconocido")


def restricciones(x, problema_id):
    if problema_id == 0:
        g = np.zeros(2)
        g[0] = (x[0] - 5) ** 2 + x[1] ** 2 - 25
        g[1] = 7.7 - ((x[0] - 8) ** 2 + (x[1] + 3) ** 2)
        return g

    elif problema_id == 3:
        g = np.zeros(2)
        g[0] = x[0] + x[1] - 400.0
        g[1] = 2.0 * x[0] + x[1] - 500.0
        return g

    else:
        return np.array([])  # sin restricciones


def obtener_parametros(problema_id):
    if problema_id == 0:
        D = 2
        limites_inf = np.array([0.0, 0.0])
        limites_sup = np.array([5.0, 3.0])
        return D, limites_inf, limites_sup

    elif problema_id == 1:
        D = 1
        limites_inf = np.array([-100000.0])
        limites_sup = np.array([100000.0])
        return D, limites_inf, limites_sup

    elif problema_id == 2:
        D = 2
        limites_inf = np.array([-5.0, -3.0])
        limites_sup = np.array([5.0, 3.0])
        return D, limites_inf, limites_sup

    elif problema_id == 3:
        D = 2
        limites_inf = np.array([0.0, 0.0])
        limites_sup = np.array([500.0, 500.0])
        return D, limites_inf, limites_sup

    else:
        raise ValueError("ID de problema no reconocido")
