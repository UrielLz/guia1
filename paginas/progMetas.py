import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from funciones2 import funciones, restricciones, obtener_parametros

np.random.seed(15)

# ====== Metas fijas por problema ======
METAS = {
    0: [0.0, 4.0],
    1: [0, 0],
    2: [-3, -5],
    3: [-130.0, -250.0]
}

# ========= FUNCIONES DEL MÉTODO GP CON AMBAS DESVIACIONES =========

def calcular_objetivo(x, problema_id, metas, pesos_p, pesos_n):
    f1, f2 = funciones(x, problema_id)
    f = [f1, f2]
    p = [max(0, metas[i] - f[i]) for i in range(2)]  # desviación por debajo
    n = [max(0, f[i] - metas[i]) for i in range(2)]  # desviación por encima
    return sum(pesos_p[i] * p[i] + pesos_n[i] * n[i] for i in range(2))

def calcular_incumplimiento(x, problema_id):
    r = restricciones(x, problema_id)
    return np.sum([ri for ri in r if ri > 0]) if len(r) > 0 else 0

def crear_poblacion(NP, D, lim_inf, lim_sup):
    return np.random.uniform(lim_inf, lim_sup, (NP, D))

def aplicar_limites(poblacion, lim_inf, lim_sup):
    return np.clip(poblacion, lim_inf, lim_sup)

def mutar(poblacion, F, lim_inf, lim_sup):
    NP, D = poblacion.shape
    nuevos = np.zeros_like(poblacion)
    for i in range(NP):
        indices = list(range(NP))
        indices.remove(i)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        nuevos[i] = poblacion[r3] + F * (poblacion[r1] - poblacion[r2])
    return aplicar_limites(nuevos, lim_inf, lim_sup)

def cruzar(poblacion, mutados, CR):
    NP, D = poblacion.shape
    hijos = np.zeros_like(poblacion)
    for i in range(NP):
        punto = np.random.randint(0, D)
        for j in range(D):
            if np.random.rand() < CR or j == punto:
                hijos[i, j] = mutados[i, j]
            else:
                hijos[i, j] = poblacion[i, j]
    return hijos

def seleccionar(poblacion, hijos, problema_id, metas, pesos_p, pesos_n):
    NP = poblacion.shape[0]
    nueva = np.zeros_like(poblacion)
    for i in range(NP):
        inc_padre = calcular_incumplimiento(poblacion[i], problema_id)
        inc_hijo = calcular_incumplimiento(hijos[i], problema_id)

        if inc_hijo == 0 and inc_padre == 0:
            obj_padre = calcular_objetivo(poblacion[i], problema_id, metas, pesos_p, pesos_n)
            obj_hijo = calcular_objetivo(hijos[i], problema_id, metas, pesos_p, pesos_n)
            nueva[i] = hijos[i] if obj_hijo < obj_padre else poblacion[i]
        elif inc_hijo == 0:
            nueva[i] = hijos[i]
        elif inc_padre == 0:
            nueva[i] = poblacion[i]
        else:
            nueva[i] = hijos[i] if inc_hijo < inc_padre else poblacion[i]
    return nueva

def resolver_GP(problema_id, D, lim_inf, lim_sup, metas, pesos_p, pesos_n, NP, F, CR, GEN):
    poblacion = crear_poblacion(NP, D, lim_inf, lim_sup)
    for _ in range(GEN):
        mutados = mutar(poblacion, F, lim_inf, lim_sup)
        cruzados = cruzar(poblacion, mutados, CR)
        poblacion = seleccionar(poblacion, cruzados, problema_id, metas, pesos_p, pesos_n)
    mejor = min(
        poblacion,
        key=lambda x: (
            calcular_incumplimiento(x, problema_id),
            calcular_objetivo(x, problema_id, metas, pesos_p, pesos_n)
        )
    )
    return mejor, funciones(mejor, problema_id)


# ================= INTERFAZ =================

def mostrar():
    st.set_page_config(layout="wide")
    st.title("Programación por Metas")

    col1, col2 = st.columns(2)

    with st.sidebar:
        st.header("Configuración")
        problema_id = st.selectbox("Selecciona el problema", [0, 1, 2, 3])
        NP = st.slider("Tamaño de población", 10, 200, 30, step=10)
        F = st.slider("Factor de mutación (F)", 0.1, 1.0, 0.5, step=0.05)
        CR = st.slider("Probabilidad de cruce (CR)", 0.0, 1.0, 0.8, step=0.05)
        GEN = st.slider("Número de generaciones", 10, 1000, 100, step=10)
        num_pesos = st.slider("Puntos del frente (número de pesos)", 10, 200, 20, step=10)


    D, lim_inf, lim_sup = obtener_parametros(problema_id)
    metas = METAS[problema_id]

    # Bucle para barrer pesos
    pesos_list = np.linspace(0, 1, num_pesos)
    pareto = []

    for w1 in pesos_list:
        w2 = 1.0 - w1
        pesos_p = [w1, w2]
        pesos_n = [w1, w2]
        x, obj = resolver_GP(problema_id, D, lim_inf, lim_sup, metas, pesos_p, pesos_n, NP, F, CR, GEN)
        if calcular_incumplimiento(x, problema_id) < 1e-5:
            pareto.append(obj)


    pareto = np.array(pareto)

    st.subheader("Frente generado con Programación por Metas")
    fig, ax = plt.subplots()
    if len(pareto) > 0:
        idx = np.argsort(pareto[:, 0])
        ax.scatter(pareto[:, 0], pareto[:, 1], c='purple', s=20, label=f"Soluciones = {len(pareto)}")
        ax.plot(pareto[idx, 0], pareto[idx, 1], linestyle='--', linewidth=1, color='magenta')
    ax.axvline(x=metas[0], linestyle=':', color='gray', label="Meta τ₁")
    ax.axhline(y=metas[1], linestyle=':', color='gray', label="Meta τ₂")
    ax.set_xlabel("f1(x)")
    ax.set_ylabel("f2(x)")
    ax.set_title(f"Frente de Pareto - Problema {problema_id}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    with col1:
        st.subheader("¿Qué es la Programación por Metas?")

        st.markdown("""
    La **Programación por Metas (Goal Programming)** es una técnica de optimización multiobjetivo que transforma el problema original en uno monoobjetivo. En lugar de minimizar directamente las funciones objetivo, se minimizan las desviaciones respecto a un conjunto de metas fijas $\\tau_k$.
        """)

        st.markdown("Se introducen variables de desviación no negativas:")

        st.latex(r"p_k \geq 0 \quad \text{(desviación por debajo de la meta } \tau_k \text{)}")
        st.latex(r"n_k \geq 0 \quad \text{(desviación por encima de la meta } \tau_k \text{)}")

        st.markdown("Estas variables se integran a la formulación mediante la siguiente ecuación:")
        st.latex(r"f_k(x) + p_k - n_k = \tau_k, \quad \text{para } k = 1, \ldots, K")

        st.markdown("**Así:**")
        st.markdown(r"- Si $f_k(x) < \tau_k$, entonces $p_k = \tau_k - f_k(x) > 0$ y $n_k = 0$")
        st.markdown(r"- Si $f_k(x) > \tau_k$, entonces $n_k = f_k(x) - \tau_k > 0$ y $p_k = 0$")
        st.markdown(r"- Si $f_k(x) = \tau_k$, entonces $p_k = n_k = 0$")

        st.markdown("El objetivo del modelo es minimizar la suma ponderada de desviaciones no deseadas:")

        st.latex(r"Z = \sum_{k=1}^K \left( w_{1,k} \cdot p_k + w_{2,k} \cdot n_k \right)")

        st.markdown("**Donde:**")
        st.markdown(r"- $w_{1,k}$: peso asignado a la desviación por debajo de la meta $\tau_k$")
        st.markdown(r"- $w_{2,k}$: peso asignado a la desviación por encima de la meta $\tau_k$")

        st.markdown("**El problema completo queda formulado como:**")

        st.latex(r"""
    \begin{aligned}
    \text{Minimizar} \quad & Z = \sum_{k=1}^K (w_{1,k} \cdot p_k + w_{2,k} \cdot n_k) \\
    \text{sujeto a} \quad & f_k(x) + p_k - n_k = \tau_k, \quad k = 1, \ldots, K \\
    & p_k \geq 0, \quad n_k \geq 0, \quad k = 1, \ldots, K \\
    & g_j(x) \leq 0, \quad j = 1, \ldots, m
    \end{aligned}
    """)

        st.markdown("**Nota:** Una **meta** $\\tau_k$ representa el valor deseado o esperado que se busca alcanzar para la función objetivo $f_k(x)$.")

    with col2:
        st.subheader("Algoritmo Paso a Paso")

        st.markdown("### 1. Definir metas fijas $\\tau_k$")
        st.code("metas = [τ₁, τ₂]  # metas para f₁(x) y f₂(x)")

        st.markdown("### 2. Recorrer pesos $w_k$")
        st.markdown(r"""
    Para generar el frente de Pareto, se recorren combinaciones de pesos:

    - $w_k = [w_1, w_2]$  
    - Cumple: $w_1 + w_2 = 1$

    Estos pesos se aplican **tanto a las desviaciones positivas ($p_k$)** como a las negativas ($n_k$):
        """)
        st.code('''
    pesos_p = [w1, w2]  # pesos para desviaciones por debajo de la meta
    pesos_n = [w1, w2]  # pesos para desviaciones por encima de la meta
        ''')

        st.markdown("### 3. Inicializar población aleatoria")
        st.code("poblacion = crear_poblacion(NP, D, lim_inf, lim_sup)")

        st.markdown("### 4. Optimizar usando DE con Reglas de Deb")
        st.code('''
    for _ in range(GEN):
        mutados = mutar(poblacion, F, lim_inf, lim_sup)
        cruzados = cruzar(poblacion, mutados, CR)
        poblacion = seleccionar(poblacion, cruzados, problema_id, metas, pesos_p, pesos_n)
        ''')

        st.markdown("### 5. Evaluar función objetivo con desviaciones")
        st.latex(r"Z(x) = \sum_{k=1}^2 \left( w_{1,k} \cdot p_k + w_{2,k} \cdot n_k \right)")

        st.markdown("Donde:")
        st.markdown(r"""
    - $p_k = \max(0, \tau_k - f_k(x))$: desviación por debajo de la meta  
    - $n_k = \max(0, f_k(x) - \tau_k)$: desviación por encima de la meta  
    - $w_{1,k}$ y $w_{2,k}$ son los pesos asignados a cada desviación
        """)

        st.code('''
    mejor = min(poblacion, key=lambda x: (
        calcular_incumplimiento(x, problema_id),
        calcular_objetivo(x, problema_id, metas, pesos_p, pesos_n)
    ))
        ''')

