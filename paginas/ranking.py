import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from random import random
from funciones import funcion_objetivo, restricciones, obtener_parametros

PALETA = {
    "linea": "#6C63FF",
    "mejor_punto": "#FF4081",
    "factibles": "#00E676",
    "no_factibles": "#FF1744",
    "optimo": "#40C4FF"
}

def mostrar():
    st.set_page_config(layout="wide")
    st.markdown("""
    <div style='background-color:#1E1E1E;padding:20px;border-radius:10px'>
        <h1 style='color:white;text-align:center;'>Evolución Diferencial con Stochastic Ranking (SR)</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    st.sidebar.title(" Configuración del problema ")

    problemas_benchmark = {
        "g01": 0, "g02": 1, "g03": 2, "g04": 3, "g05": 4, "g06": 5, "g07": 6,
        "g08": 7, "g09": 8, "g10": 9, "g11": 10, "g12": 11, "g13": 12, "g14": 13,
        "g15": 14, "g16": 15, "g17": 16, "g18": 17, "g19": 18, "g20": 19, "g21": 20,
        "g22": 21, "g23": 22, "g24": 23
    }

    opcion_problema = st.sidebar.selectbox("Selecciona un problema:", list(problemas_benchmark.keys()))
    pid = problemas_benchmark[opcion_problema]
    D, lim_inf_vec, lim_sup_vec = obtener_parametros(pid)

    NP = st.sidebar.slider("Tamaño de población (NP)", 10, 100, 30, 5)
    F = st.sidebar.slider("Factor de mutación (F)", 0.0, 1.0, 0.5)
    CR = st.sidebar.slider("Tasa de cruce (CR)", 0.0, 1.0, 0.9)
    generaciones = st.sidebar.slider("Generaciones", 10, 200, 50, step=10)
    Pf = st.sidebar.slider("Probabilidad Pf", 0.0, 1.0, 0.45)
    mostrar_optimo = st.sidebar.checkbox("Marcar mejor solución", value=True)

    def calcular_incumplimiento(x, pid):
        return np.sum(np.maximum(0, restricciones(x, pid)))

    def seleccion_sr_original(padres, hijos, problema_id):
        combinada = np.vstack((padres, hijos))
        tam_total = 2 * NP

        f_obj = np.array([funcion_objetivo(sol, problema_id) for sol in combinada])
        phi_vals = np.array([calcular_incumplimiento(sol, problema_id) for sol in combinada])

        for n in range(tam_total):
            for j in range(tam_total - 1):
                u = random()
                phi1, phi2 = phi_vals[j], phi_vals[j+1]
                f1, f2 = f_obj[j], f_obj[j+1]
                ambos_factibles = (phi1 <= 1e-6) and (phi2 <= 1e-6)
                intercambiar = False

                if ambos_factibles or u < Pf:
                    if f1 > f2:
                        intercambiar = True
                else:
                    if phi1 > phi2:
                        intercambiar = True

                if intercambiar:
                    combinada[[j, j+1]] = combinada[[j+1, j]]
                    f_obj[[j, j+1]] = f_obj[[j+1, j]]
                    phi_vals[[j, j+1]] = phi_vals[[j+1, j]]

        return combinada[:NP, :]

    def evolucion_diferencial_sr():
        mejor_sol = None
        mejor_valor = float("inf")
        poblacion = lim_inf_vec + np.random.rand(NP, D) * (lim_sup_vec - lim_inf_vec)
        historial = []

        for gen in range(generaciones):
            mutados = np.zeros_like(poblacion)
            for i in range(NP):
                idxs = list(range(NP))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                v = poblacion[r3] + F * (poblacion[r1] - poblacion[r2])
                v = np.clip(v, lim_inf_vec, lim_sup_vec)
                mutados[i] = v

            ensayo = np.zeros_like(poblacion)
            for i in range(NP):
                k = np.random.randint(D)
                for j in range(D):
                    if np.random.rand() <= CR or j == k:
                        ensayo[i, j] = mutados[i, j]
                    else:
                        ensayo[i, j] = poblacion[i, j]

            poblacion = seleccion_sr_original(poblacion, ensayo, pid)

            factibles = [funcion_objetivo(ind, pid) for ind in poblacion if calcular_incumplimiento(ind, pid) <= 1e-6]
            historial.append(min(factibles) if factibles else None)

            for ind in poblacion:
                if calcular_incumplimiento(ind, pid) <= 1e-6:
                    val = funcion_objetivo(ind, pid)
                    if val < mejor_valor:
                        mejor_valor = val
                        mejor_sol = ind.copy()

        return historial, mejor_sol, mejor_valor

    historial, mejor_sol, mejor_valor = evolucion_diferencial_sr()

    st.subheader(" Evolución de la solución")
    fig, ax = plt.subplots()
    valores_limpios = [v for v in historial if v is not None]
    ax.plot(valores_limpios, color=PALETA["linea"], label="f(x) mínimo")
    if mostrar_optimo and valores_limpios:
        min_gen = np.argmin(valores_limpios)
        ax.plot(min_gen, valores_limpios[min_gen], 'o', color=PALETA["mejor_punto"], label="Mejor f(x)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("f(x)")
    ax.set_title("Optimización con SR")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    if mejor_sol is not None:
        st.success(f" Mejor solución encontrada: x = {np.round(mejor_sol, 4).tolist()}, f(x) = {mejor_valor:.6f}")

    bloques = {
        "Inicialización": """
def inicializar_poblacion():
    return lim_inf + np.random.rand(NP, dim) * (lim_sup - lim_inf)
""",
        "Mutación": """
def mutacion(poblacion):
    mutados = np.zeros_like(poblacion)
    for i in range(NP):
        idxs = list(range(NP))
        idxs.remove(i)
        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
        v = poblacion[r3] + F * (poblacion[r1] - poblacion[r2])
        mutados[i] = np.clip(v, lim_inf, lim_sup)
    return mutados
""",
        "Cruce": """
def cruce(poblacion, mutados):
    trial = np.zeros_like(poblacion)
    for i in range(NP):
        k = np.random.randint(dim)
        for j in range(dim):
            if np.random.rand() <= CR or j == k:
                trial[i, j] = mutados[i, j]
            else:
                trial[i, j] = poblacion[i, j]
    return trial
""",
        "Selección": """
def seleccion_sr_original(padres, hijos, problema_id):
    combinada = np.vstack((padres, hijos))
    tam_total = 2 * NP
    f_obj = np.array([funcion_objetivo(sol, problema_id) for sol in combinada])
    phi_vals = np.array([calcular_incumplimiento(sol, problema_id) for sol in combinada])

    for n in range(tam_total):
        for j in range(tam_total - 1):
            u = random()
            phi1, phi2 = phi_vals[j], phi_vals[j+1]
            f1, f2 = f_obj[j], f_obj[j+1]
            ambos_factibles = (phi1 <= 1e-6) and (phi2 <= 1e-6)
            intercambiar = False

            if ambos_factibles or u < Pf:
                if f1 > f2:
                    intercambiar = True
            else:
                if phi1 > phi2:
                    intercambiar = True

            if intercambiar:
                combinada[[j, j+1]] = combinada[[j+1, j]]
                f_obj[[j, j+1]] = f_obj[[j+1, j]]
                phi_vals[[j, j+1]] = phi_vals[[j+1, j]]

    return combinada[:NP, :]
"""
    }

    with col1:
        st.subheader(" Pseudocódigo ")
        opcion = st.radio("Selecciona un bloque:", list(bloques.keys()), horizontal=True)

        if opcion == "Inicialización":
            st.latex(r"x_{j,i,0} = rnd_j[0,1] \cdot (x_j^U - x_j^L) + x_j^L")
        elif opcion == "Mutación":
            st.latex(r"v_{j,i,G+1} = x_{j,r3,G} + F \cdot (x_{j,r1,G} - x_{j,r2,G})")
        elif opcion == "Cruce":
            st.latex(r"u_{j,i,G+1} = \begin{cases} v_{j,i,G+1} & \text{si } rnd[0,1] \leq CR \lor j = k \\ x_{j,i,G} & \text{en otro caso} \end{cases}")
        elif opcion == "Selección":
            st.latex(r"x_i^{G+1} = \begin{cases} u_i & \text{si } \phi(u_i) \leq \varepsilon \text{ y es mejor (Pf)} \\ x_i & \text{en otro caso} \end{cases}")
            st.markdown(r"""
            El algoritmo **Stochastic Ranking** decide si un individuo de prueba $u_i$ reemplaza al actual $x_i$ con base en una combinación probabilística entre:

            - **Función objetivo** $f(x)$ (qué tan buena es la solución).
            - **Función de penalización** $\phi(x)$ (qué tanto incumple las restricciones).

            #### Variables:
            - **$x_i^{G+1}$**: Individuo seleccionado para la siguiente generación.
            - **$u_i$**: Individuo de prueba (hijo).
            - **$x_i$**: Individuo actual (padre).
            - **$\phi(x)$**: Suma de incumplimientos de las restricciones (penalización).
            - **$\varepsilon$**: Umbral de factibilidad tolerable.
            - **Pf**: Probabilidad de usar $f(x)$ como criterio de comparación.

            #### Reglas del algoritmo:
            1. Si **ambos individuos son factibles** (es decir, $\phi(u_i) \approx 0$ y $\phi(x_i) \approx 0$), o con **probabilidad Pf**, se comparan por función objetivo: se prefiere el que tenga **menor $f(x)$**.
            2. En otro caso, se comparan por penalización $\phi(x)$: se prefiere el que **viole menos las restricciones**.

            """)

    with col2:
        st.subheader(f" Paso de {opcion}")
        st.code(bloques[opcion], language="python")
