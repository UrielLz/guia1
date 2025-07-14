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
        "<h1 style='text-align: center; color: white; background-color: #1E1E1E; padding: 1rem; border-radius: 0.5rem;'>Grey Wolf Optimizer (GWO)</h1>",
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

    def ejecutar_gwo():
        X = lim_inf + np.random.rand(NP, D) * (lim_sup - lim_inf)
        alpha = np.zeros(D)
        beta = np.zeros(D)
        delta = np.zeros(D)

        f_alpha = float("inf")
        f_beta = float("inf")
        f_delta = float("inf")
        phi_alpha = phi_beta = phi_delta = float("inf")

        historial = []

        for iter in range(generaciones):
            for i in range(NP):
                fit = funcion_objetivo(X[i], pid)
                incum = phi(X[i])

                if incum < phi_alpha or (incum <= TOL and fit < f_alpha):
                    delta, f_delta, phi_delta = beta.copy(), f_beta,phi_beta
                    beta, f_beta, phi_beta = alpha.copy(), f_alpha, phi_alpha
                    alpha, f_alpha, phi_alpha = X[i].copy(), fit, incum
                elif incum < phi_beta or (incum <= TOL and fit < f_beta):
                    delta, f_delta, phi_delta = beta.copy(), f_beta, phi_beta
                    beta, f_beta, phi_beta = X[i].copy(), fit, incum
                elif incum < phi_delta or (incum <= TOL and fit < f_delta):
                    delta, f_delta, phi_delta = X[i].copy(), fit, incum

            a = 2 - iter * (2 / generaciones)

            for i in range(NP):
                for j in range(D):
                    # Alpha
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - X[i, j])
                    X1 = alpha[j] - A1 * D_alpha

                    # Beta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - X[i, j])
                    X2 = beta[j] - A2 * D_beta

                    # Delta
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - X[i, j])
                    X3 = delta[j] - A3 * D_delta

                    # Posición nueva
                    X[i, j] = np.clip((X1 + X2 + X3) / 3, lim_inf[j], lim_sup[j])

            factibles = [funcion_objetivo(x, pid) for x in X if phi(x) <= TOL]
            historial.append(min(factibles) if factibles else None)

        return historial, alpha, f_alpha

    # Ejecutar
    historial, mejor_sol, mejor_val = ejecutar_gwo()

    # Gráfica
    st.subheader("Evolución de la solución")
    fig, ax = plt.subplots()
    valores_limpios = [v for v in historial if v is not None]
    ax.plot(valores_limpios, color=PALETA["linea"], label="f(x) mínimo")
    if mostrar_optimo and valores_limpios:
        idx = np.argmin(valores_limpios)
        ax.plot(idx, valores_limpios[idx], 'o', color=PALETA["mejor_punto"], label="Mejor f(x)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("f(x)")
    ax.set_title("Optimización con GWO")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    if mejor_sol is not None:
        st.success(f"Mejor solución encontrada: x = {np.round(mejor_sol, 4).tolist()}, f(x) = {mejor_val:.6f}")

    # Bloques visuales como en las otras páginas
    bloques = {
        "Inicialización": {
            "exp": r"""
$$
x_{j,i,0} = rnd_j[0,1] \cdot (x_j^U - x_j^L) + x_j^L
$$
Esta fórmula genera la **población inicial** de individuos distribuidos aleatoriamente dentro del espacio de búsqueda.

### Variables:
- $x_{j,i,0}$: Valor de la variable $j$ del individuo $i$ en la generación 0.
- $rnd_j[0,1]$: Número aleatorio uniforme en $[0,1]$ específico para la dimensión $j$.
- $x_j^U$, $x_j^L$: Límites superior e inferior del dominio de la variable $x_j$.

La fórmula asegura que **cada componente de cada individuo se inicialice dentro de su dominio permitido**.
""",
            "code": '''# Inicialización de la población
X = lim_inf + np.random.rand(NP, D) * (lim_sup - lim_inf)

# Variables de líderes
alpha = np.zeros(D)
beta = np.zeros(D)
delta = np.zeros(D)

f_alpha = float("inf")
f_beta = float("inf")
f_delta = float("inf")

phi_alpha = phi_beta = phi_delta = float("inf")

historial = []'''
        },
        "Evaluación": {
            "exp": r"""
$$
\phi(x) = \sum_{j=1}^{m} \max(0, g_j(x))
$$
Esta expresión calcula la **violación total de restricciones** de una solución $x$.

### Variables:
- $\phi(x)$: Penalización total de la solución.
- $g_j(x)$: Evaluación de la j-ésima restricción.
- $\max(0, g_j(x))$: Solo se penaliza si $g_j(x) > 0$ (restricción incumplida).
""",
            "code": '''# Función de penalización por restricciones
def phi(x):
    return np.sum(np.maximum(0, restricciones(x, pid)))

# Evaluación de la población
def evaluar_poblacion():
    for i in range(NP):
        fit = funcion_objetivo(X[i], pid)
        incum = phi(X[i])
        yield i, fit, incum'''
        },
        "Actualización de líderes": {
            "exp": r"""
Se seleccionan los tres mejores lobos: $\alpha$, $\beta$, y $\delta$ usando las **reglas de Deb**:

1. Si $\phi(x_1) < \phi(x_2)$: se prefiere $x_1$
2. Si $\phi(x_1) = \phi(x_2)$ y ambas son factibles, se prefiere la que tenga menor $f(x)$

Esto asegura un equilibrio entre factibilidad y calidad.

### Variables:
- $\phi(x)$: Penalización (violación de restricciones).
- $f(x)$: Valor de la función objetivo.
- $\alpha$, $\beta$, $\delta$: Mejores tres soluciones de la población actual.
""",
            "code": '''# Selección de líderes alpha, beta y delta según reglas de Deb
for i, fit, incum in evaluar_poblacion():
    if incum < phi_alpha or (incum <= TOL and fit < f_alpha):
        delta, f_delta, phi_delta = beta.copy(), f_beta, phi_beta
        beta, f_beta, phi_beta = alpha.copy(), f_alpha, phi_alpha
        alpha, f_alpha, phi_alpha = X[i].copy(), fit, incum
    elif incum < phi_beta or (incum <= TOL and fit < f_beta):
        delta, f_delta, phi_delta = beta.copy(), f_beta, phi_beta
        beta, f_beta, phi_beta = X[i].copy(), fit, incum
    elif incum < phi_delta or (incum <= TOL and fit < f_delta):
        delta, f_delta, phi_delta = X[i].copy(), fit, incum'''
        },
        "Actualización de posiciones": {
            "exp": r"""
Cada lobo actualiza su posición guiado por $\alpha$, $\beta$ y $\delta$ usando las siguientes fórmulas:

$$
\begin{aligned}
A_k &= 2a \cdot r_1 - a \\
C_k &= 2 \cdot r_2 \\
D_k &= |C_k \cdot X_k - X| \\
X_k' &= X_k - A_k \cdot D_k \\
X_{new} &= \frac{X_1' + X_2' + X_3'}{3}
\end{aligned}
$$

### Variables:
- $a$: parámetro que decrece linealmente de 2 a 0.
- $r_1$, $r_2$: números aleatorios ∈ [0, 1].
- $X_k$: posición de $\alpha$, $\beta$ o $\delta$.
- $X$: posición actual del lobo.
- $X_{new}$: nueva posición promedio.
""",
            "code": '''# Actualización de la posición de los lobos
a = 2 - iter * (2 / generaciones)

for i in range(NP):
    for j in range(D):
        # Alpha
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha[j] - X[i, j])
        X1 = alpha[j] - A1 * D_alpha

        # Beta
        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta[j] - X[i, j])
        X2 = beta[j] - A2 * D_beta

        # Delta
        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta[j] - X[i, j])
        X3 = delta[j] - A3 * D_delta

        # Promedio de las posiciones
        X[i, j] = np.clip((X1 + X2 + X3) / 3, lim_inf[j], lim_sup[j])'''
        }
    }

    with col1:
        st.subheader("Pseudocódigo")
        paso = st.radio("Selecciona un bloque:", list(bloques.keys()), horizontal=True)
        st.markdown(bloques[paso]['exp'], unsafe_allow_html=True)

    with col2:
        st.subheader(f"Código del paso: {paso}")
        st.code(bloques[paso]["code"], language="python")
