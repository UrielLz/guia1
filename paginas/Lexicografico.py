import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from funciones2 import funciones, restricciones, obtener_parametros

np.random.seed(15)

def mostrar():

    
    col1, col2 = st.columns(2)

    st.set_page_config(layout="wide")
    st.title("Optimización Lexicográfica")

    with st.sidebar:
        st.header("Configuración")
        problema_id = st.selectbox("Selecciona el problema", [0, 1, 2, 3])
        NP = st.slider("Tamaño de población", 10, 200, 30, step=10)
        F = st.slider("Factor de mutación (F)", 0.1, 1.0, 0.6, step=0.05)
        CR = st.slider("Probabilidad de cruce (CR)", 0.0, 1.0, 0.9, step=0.05)
        GEN = st.slider("Número de generaciones", 10, 1000, 100, step=10)
        num_alphas = st.slider("Número de alphas", 10, 200, 20, step=10)
        alpha_min = st.number_input("Alpha mínimo", value=-50.0)
        alpha_max = st.number_input("Alpha máximo", value=50.0)

    D, lim_inf, lim_sup = obtener_parametros(problema_id)

    def crear_poblacion():
        return np.random.uniform(lim_inf, lim_sup, (NP, D))

    def aplicar_limites(pob):
        return np.clip(pob, lim_inf, lim_sup)

    def mutar(pob):
        hijos = np.zeros_like(pob)
        for i in range(NP):
            r1, r2, r3 = np.random.choice(np.delete(np.arange(NP), i), 3, replace=False)
            hijos[i] = pob[r3] + F * (pob[r1] - pob[r2])
        return aplicar_limites(hijos)

    def cruzar(pob, mutados):
        hijos = np.zeros_like(pob)
        for i in range(NP):
            punto = np.random.randint(0, D)
            for j in range(D):
                hijos[i, j] = mutados[i, j] if np.random.rand() < CR or j == punto else pob[i, j]
        return hijos

    def calcular_incumplimiento(x):
        return np.sum([r for r in restricciones(x, problema_id) if r > 0])

    def calcular_incumplimiento_f1(x, limite_f1, tol=1e-4):
        inc = calcular_incumplimiento(x)
        f1_val = funciones(x, problema_id)[0]
        inc_f1 = max(0, f1_val - (limite_f1 + tol))
        return inc + 1e6 * inc_f1

    def seleccionar_min_f1(pob, hijos):
        nueva = np.zeros_like(pob)
        for i in range(NP):
            p, h = pob[i], hijos[i]
            f1_p, f1_h = funciones(p, problema_id)[0], funciones(h, problema_id)[0]
            inc_p, inc_h = calcular_incumplimiento(p), calcular_incumplimiento(h)
            if inc_p == 0 and inc_h == 0:
                nueva[i] = h if f1_h < f1_p else p
            elif inc_h == 0:
                nueva[i] = h
            elif inc_p == 0:
                nueva[i] = p
            else:
                nueva[i] = h if inc_h < inc_p else p
        return nueva

    def seleccionar_min_f2_restringido(pob, hijos, limite_f1):
        nueva = np.zeros_like(pob)
        for i in range(NP):
            p, h = pob[i], hijos[i]
            f2_p, f2_h = funciones(p, problema_id)[1], funciones(h, problema_id)[1]
            inc_p = calcular_incumplimiento_f1(p, limite_f1)
            inc_h = calcular_incumplimiento_f1(h, limite_f1)
            if inc_p == 0 and inc_h == 0:
                nueva[i] = h if f2_h < f2_p else p
            elif inc_h == 0:
                nueva[i] = h
            elif inc_p == 0:
                nueva[i] = p
            else:
                nueva[i] = h if inc_h < inc_p else p
        return nueva

    def resolver_min_f1():
        pob = crear_poblacion()
        for _ in range(GEN):
            mutados = mutar(pob)
            cruzados = cruzar(pob, mutados)
            pob = seleccionar_min_f1(pob, cruzados)
        mejor = min(pob, key=lambda x: (calcular_incumplimiento(x), funciones(x, problema_id)[0]))
        return mejor, funciones(mejor, problema_id)[0]

    def resolver_min_f2(limite_f1):
        pob = crear_poblacion()
        for _ in range(GEN):
            mutados = mutar(pob)
            cruzados = cruzar(pob, mutados)
            pob = seleccionar_min_f2_restringido(pob, cruzados, limite_f1)
        mejor = min(pob, key=lambda x: (calcular_incumplimiento_f1(x, limite_f1), funciones(x, problema_id)[1]))
        return mejor, funciones(mejor, problema_id)

   
    x_f1_opt, f1_opt = resolver_min_f1()


    alpha_list = np.linspace(alpha_min, alpha_max, num_alphas)
    pareto = []

    for i, alpha in enumerate(alpha_list):
        limite_f1 = f1_opt + alpha
        x, objetivos = resolver_min_f2(limite_f1)
        if calcular_incumplimiento(x) < 1e-6:
            pareto.append(objetivos)

    pareto = np.array(pareto)
    st.subheader("Frente Lexicográfico")
    fig, ax = plt.subplots()
    if len(pareto) > 0:
        idx = np.argsort(pareto[:, 0])
        ax.scatter(pareto[:, 0], pareto[:, 1], c='purple', s=20, label=f"Soluciones = {len(pareto)}")
        ax.plot(pareto[idx, 0], pareto[idx, 1], linestyle='--', color='magenta', linewidth=1)
    ax.set_xlabel("f1(x)")
    ax.set_ylabel("f2(x)")
    ax.set_title(f"Frente Lexicográfico - Problema {problema_id}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    
    with col1:
        st.subheader("¿Qué es la Optimización Lexicográfica?")

        st.markdown("""
    La **Optimización Lexicográfica** es una técnica que resuelve problemas multiobjetivo **priorizando los objetivos** por orden de importancia.

    El procedimiento se realiza en dos fases:
    """)

        st.markdown("### Fase 1: Prioridad al primer objetivo $f_1(x)$")
        st.markdown("""
    Se **minimiza $f_1(x)$** sujeto únicamente a las restricciones originales del problema.
    """)

        st.latex(r"""
    \begin{aligned}
    \text{Minimizar} \quad & f_1(x) \\
    \text{sujeto a} \quad & g_j(x) \leq 0, \quad j = 1, \ldots, m
    \end{aligned}
    """)

        st.markdown("El resultado es el valor óptimo mínimo \\( f_1^* \\).")

        st.markdown("### Fase 2: Minimizar $f_2(x)$ con restricción en $f_1(x)$")
        st.markdown("""
    Se busca minimizar $f_2(x)$, pero **manteniendo $f_1(x)$ cerca de su valor mínimo** encontrado en la Fase 1.  
    Esto se hace agregando una restricción relajada por un parámetro \\( \\alpha \\geq 0 \\):
    """)

        st.latex(r"f_1(x) \leq f_1^* + \alpha")

        st.markdown("El modelo queda así:")

        st.latex(r"""
    \begin{aligned}
    \text{Minimizar} \quad & f_2(x) \\
    \text{sujeto a} \quad & f_1(x) \leq f_1^* + \alpha \\
                          & g_j(x) \leq 0, \quad j = 1, \ldots, m
    \end{aligned}
    """)

        st.markdown("""
    Al variar \\( \\alpha \\) en un rango dado, se obtiene un conjunto de soluciones factibles
    que forman una **aproximación al frente de Pareto**, pero respetando la **preferencia jerárquica** de los objetivos.
    """)

    with col2:
        st.subheader("Algoritmo Paso a Paso")

        st.markdown("### 1. Inicializar población aleatoria")
        st.code("poblacion = crear_poblacion()")

        st.markdown("### 2. Minimizar $f_1(x)$ con restricciones originales")
        st.markdown("Se ejecuta DE para obtener el valor óptimo mínimo:")
        st.code('''
for _ in range(GEN):
    mutados = mutar(poblacion)
    cruzados = cruzar(poblacion, mutados)
    poblacion = seleccionar_min_f1(poblacion, cruzados)

mejor = min(poblacion, key=lambda x: (calcular_incumplimiento(x), funciones(x, problema_id)[0]))
f1_opt = funciones(mejor, problema_id)[0]
        ''')

        st.markdown("### 3. Definir rango de alphas")
        st.markdown("El valor óptimo \\( f_1^* \\) se relaja usando un parámetro \\( \\alpha \\):")
        st.latex(r"f_1(x) \leq f_1^* + \alpha")

        st.code('''
alpha_list = np.linspace(alpha_min, alpha_max, num_alphas)
        ''')

        st.markdown("### 4. Minimizar $f_2(x)$ con la nueva restricción")
        st.code('''
for alpha in alpha_list:
    limite_f1 = f1_opt + alpha
    for _ in range(GEN):
        mutados = mutar(poblacion)
        cruzados = cruzar(poblacion, mutados)
        poblacion = seleccionar_min_f2_restringido(poblacion, cruzados, limite_f1)
        ''')

        st.markdown("### 5. Evaluar factibilidad y construir el frente")
        st.code('''
mejor = min(poblacion, key=lambda x: (
    calcular_incumplimiento_f1(x, limite_f1),
    funciones(x, problema_id)[1]
))
if calcular_incumplimiento(x) < 1e-6:
    pareto.append(funciones(mejor, problema_id))
        ''')

        st.success("Al recorrer distintos valores de \\( \\alpha \\), se obtiene un frente de soluciones factibles")
