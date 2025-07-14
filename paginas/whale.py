import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    st.markdown(
        """
        <h1 style='text-align: center; color: white; background-color: #1E1E1E; padding: 1rem; border-radius: 0.5rem;'>Whale Optimization Algorithm (WOA)</h1>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    st.sidebar.title("Configuración del problema")

    problemas_benchmark = {f"g{str(i+1).zfill(2)}": i for i in range(24)}
    opcion_problema = st.sidebar.selectbox("Selecciona un problema:", list(problemas_benchmark.keys()))
    pid = problemas_benchmark[opcion_problema]
    D, lim_inf, lim_sup = obtener_parametros(pid)

    NP = st.sidebar.slider("Tamaño de población (NP)", 10, 100, 30, 5)
    generaciones = st.sidebar.slider("Generaciones", 10, 200, 50, step=10)
    TOL = st.sidebar.number_input("Tolerancia de factibilidad", value=1e-6, format="%e")
    mostrar_optimo = st.sidebar.checkbox("Marcar mejor solución", value=True)

    def phi(x):
        return np.sum(np.maximum(0, restricciones(x, pid)))

    def ejecutar_woa(NP, D, lim_inf, lim_sup, generaciones, TOL, pid):
        X = lim_inf + np.random.rand(NP, D) * (lim_sup - lim_inf)
        X_best = None
        f_best = float("inf")
        phi_best = float("inf")

        historial = []
        b = 1  # parámetro espiral

        for t in range(generaciones):
            for i in range(NP):
                fit = funcion_objetivo(X[i], pid)
                infeas = phi(X[i])

                if infeas < phi_best or (infeas <= TOL and fit < f_best):
                    X_best = X[i].copy()
                    f_best = fit
                    phi_best = infeas

            a = 2 - t * (2 / generaciones)

            for i in range(NP):
                r1 = np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * np.random.rand()
                p = np.random.rand()
                l = np.random.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * X_best - X[i])
                        X[i] = X_best - A * D
                    else:
                        rand_idx = np.random.randint(NP)
                        X_rand = X[rand_idx]
                        D = abs(C * X_rand - X[i])
                        X[i] = X_rand - A * D
                else:
                    D = abs(X_best - X[i])
                    X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + X_best

                X[i] = np.clip(X[i], lim_inf, lim_sup)

            factibles = [funcion_objetivo(x, pid) for x in X if phi(x) <= TOL]
            historial.append(min(factibles) if factibles else None)

        return historial, X_best, f_best

    historial, mejor_sol, mejor_val = ejecutar_woa(NP, D, lim_inf, lim_sup, generaciones, TOL, pid)

    st.subheader("Evolución de la solución")
    fig, ax = plt.subplots()
    valores_limpios = [v for v in historial if v is not None]
    ax.plot(valores_limpios, color=PALETA["linea"], label="f(x) mínimo")
    if mostrar_optimo and valores_limpios:
        idx = np.argmin(valores_limpios)
        ax.plot(idx, valores_limpios[idx], 'o', color=PALETA["mejor_punto"], label="Mejor f(x)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("f(x)")
    ax.set_title("Optimización con WOA")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    if mejor_sol is not None:
        st.success(f"Mejor solución encontrada: x = {np.round(mejor_sol, 4).tolist()}, f(x) = {mejor_val:.6f}")

    historial, mejor_sol, mejor_val = ejecutar_woa(NP, D, lim_inf, lim_sup, generaciones, TOL, pid)
    
    # Bloques visuales estilo wolf.py
    bloques = {
        "Inicialización": {
            "exp": r"""
$$
x_{i,j} = x_j^L + rand() \cdot (x_j^U - x_j^L)
$$

Se genera aleatoriamente la población inicial dentro de los límites definidos.

### Variables:
- $x_{i,j}$: Posición de la variable $j$ del individuo $i$.
- $x_j^L$: Límite inferior de la variable $j$.
- $x_j^U$: Límite superior de la variable $j$.
- $rand()$: Valor aleatorio ∈ [0, 1] para distribuir uniformemente.
- $NP$: Tamaño de la población.
- $D$: Dimensión del problema (número de variables).
""",
            "code": '''# Inicialización de la población
X = lim_inf + np.random.rand(NP, D) * (lim_sup - lim_inf)'''
        },
        "Evaluación": {
            "exp": r"""
$$
\phi(x) = \sum_j \max(0, g_j(x))
$$

Función de penalización que mide la violación total de restricciones.

### Variables:
- $\phi(x)$: Grado total de incumplimiento.
- $g_j(x)$: j-ésima función de restricción (puede ser desigualdad o igualdad).
- $TOL$: Umbral bajo el cual una solución es considerada factible.
- $restricciones(x, pid)$: Función que evalúa todas las restricciones del problema seleccionado.
""",
            "code": '''# Penalización por restricciones
def phi(x):
    return np.sum(np.maximum(0, restricciones(x, pid)))'''
        },
        "Actualización del mejor": {
            "exp": r"""
Se utiliza la Regla de Deb para seleccionar la mejor solución entre los individuos evaluados.

### Criterios:
1. Menor $\phi(x)$ tiene prioridad.
2. Si $\phi(x_1) \approx \phi(x_2) \leq TOL$, se elige el de menor $f(x)$.

### Variables:
- $X[i]$: Individuo actual.
- $X^*$: Mejor solución encontrada hasta el momento.
- $f(x)$: Valor de la función objetivo.
- $\phi(x)$: Penalización asociada a $x$.
""",
            "code": '''# Reglas de Deb para actualizar la mejor solución
if infeas < phi_best or (infeas <= TOL and fit < f_best):
    X_best = X[i].copy()
    f_best = fit
    phi_best = infeas'''
        },
        "Movimiento WOA": {
            "exp": r"""
Actualización de posiciones basada en el comportamiento social de las ballenas:

### 1. Rodear la presa:
$$
X_{i}(t+1) = X^* - A \cdot |C \cdot X^* - X_i|
$$

### 2. Espiral de burbujas:
$$
X_{i}(t+1) = D \cdot e^{bl} \cdot \cos(2\pi l) + X^*
$$

### 3. Exploración:
$$
X_{i}(t+1) = X_{rand} - A \cdot |C \cdot X_{rand} - X_i|
$$

### Variables:
- $X_i$: Posición actual del individuo $i$.
- $X^*$: Mejor solución conocida.
- $X_{rand}$: Solución aleatoria de la población.
- $A = 2ar_1 - a$, $C = 2r_2$: Coeficientes adaptativos.
- $b$: Constante para la forma de espiral.
- $l$: Variable aleatoria ∈ [-1,1].
- $p$: Probabilidad ∈ [0,1] que decide el mecanismo de actualización.
""",
            "code": '''# Movimiento de las ballenas
a = 2 - t * (2 / generaciones)

for i in range(NP):
    r1 = np.random.rand()
    A = 2 * a * r1 - a
    C = 2 * np.random.rand()
    p = np.random.rand()
    l = np.random.uniform(-1, 1)

    if p < 0.5:
        if abs(A) < 1:
            D = abs(C * X_best - X[i])
            X[i] = X_best - A * D
        else:
            rand_idx = np.random.randint(NP)
            X_rand = X[rand_idx]
            D = abs(C * X_rand - X[i])
            X[i] = X_rand - A * D
    else:
        D = abs(X_best - X[i])
        X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + X_best

    # Corrección por límites
    X[i] = np.clip(X[i], lim_inf, lim_sup)'''
        }
    }

    with col1:
        st.subheader("Pseudocódigo")
        paso = st.radio("Selecciona un bloque:", list(bloques.keys()), horizontal=True)
        st.markdown(bloques[paso]['exp'], unsafe_allow_html=True)

    with col2:
        st.subheader(f"Código del paso: {paso}")
        st.code(bloques[paso]["code"], language="python")
