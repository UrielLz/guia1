import numpy as np
from funciones2 import funciones, restricciones, obtener_parametros

np.random.seed(42)

# ========== PARÁMETROS ==========
NP = 100
F = 0.5
CR = 0.9
GEN = 200

# =============================
# FUNCIONES AUXILIARES (DEB)
# =============================

def calcular_incumplimiento(x, problema_id):
    r = restricciones(x, problema_id)
    return np.sum([val for val in r if val > 0]) if len(r) > 0 else 0

def crear_poblacion(D, lim_inf, lim_sup):
    return np.random.uniform(lim_inf, lim_sup, (NP, D))

def aplicar_limites(pob, lim_inf, lim_sup):
    return np.clip(pob, lim_inf, lim_sup)

def mutar(pob, F, lim_inf, lim_sup):
    NP, D = pob.shape
    nuevos = np.zeros_like(pob)
    for i in range(NP):
        indices = list(range(NP))
        indices.remove(i)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        nuevos[i] = pob[r3] + F * (pob[r1] - pob[r2])
    return aplicar_limites(nuevos, lim_inf, lim_sup)

def cruzar(pob, mutados, CR):
    NP, D = pob.shape
    hijos = np.zeros_like(pob)
    for i in range(NP):
        punto = np.random.randint(0, D)
        for j in range(D):
            if np.random.rand() < CR or j == punto:
                hijos[i, j] = mutados[i, j]
            else:
                hijos[i, j] = pob[i, j]
    return hijos

def seleccionar(pob, hijos, idx_obj, problema_id):
    nueva = np.zeros_like(pob)
    for i in range(NP):
        fx_padre = funciones(pob[i], problema_id)[idx_obj]
        fx_hijo = funciones(hijos[i], problema_id)[idx_obj]
        inc_padre = calcular_incumplimiento(pob[i], problema_id)
        inc_hijo = calcular_incumplimiento(hijos[i], problema_id)

        if inc_hijo == 0 and inc_padre == 0:
            nueva[i] = hijos[i] if fx_hijo < fx_padre else pob[i]
        elif inc_hijo == 0:
            nueva[i] = hijos[i]
        elif inc_padre == 0:
            nueva[i] = pob[i]
        else:
            nueva[i] = hijos[i] if inc_hijo < inc_padre else pob[i]
    return nueva

def optimizar_individual(idx_obj, problema_id):
    D, lim_inf, lim_sup = obtener_parametros(problema_id)
    pob = crear_poblacion(D, lim_inf, lim_sup)
    for _ in range(GEN):
        mut = mutar(pob, F, lim_inf, lim_sup)
        cruz = cruzar(pob, mut, CR)
        pob = seleccionar(pob, cruz, idx_obj, problema_id)
    mejor = min(
        pob,
        key=lambda x: (
            calcular_incumplimiento(x, problema_id),
            funciones(x, problema_id)[idx_obj]
        )
    )
    f1, f2 = funciones(mejor, problema_id)
    return mejor, f1, f2

# ========================
# EJECUCIÓN POR PROBLEMA
# ========================
for pid in [0, 1, 2, 3]:
    print(f"\n--- Problema {pid} ---")
    x1, f1, f2 = optimizar_individual(0, pid)
    x2, f1b, f2b = optimizar_individual(1, pid)

    print("Min f1(x):", round(f1, 4), "| f2:", round(f2, 4))
    print("Min f2(x):", round(f2b, 4), "| f1:", round(f1b, 4))
