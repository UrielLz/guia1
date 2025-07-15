import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from funciones2 import funciones, restricciones, obtener_parametros

np.random.seed(15)

# =======================
# FUNCIONES AUXILIARES
# =======================

def calcular_objetivo(x, pesos, problema_id):
    f1, f2 = funciones(x, problema_id)
    return pesos[0] * f1 + pesos[1] * f2

def calcular_incumplimiento(x, problema_id):
    r = restricciones(x, problema_id)
    return np.sum([val for val in r if val > 0]) if len(r) > 0 else 0

def crear_poblacion_inicial(D, lim_inf, lim_sup, NP):
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

def seleccionar(poblacion, hijos, pesos, problema_id):
    NP = poblacion.shape[0]
    nueva_poblacion = np.zeros_like(poblacion)
    for i in range(NP):
        obj_act = calcular_objetivo(poblacion[i], pesos, problema_id)
        inc_act = calcular_incumplimiento(poblacion[i], problema_id)

        obj_hijo = calcular_objetivo(hijos[i], pesos, problema_id)
        inc_hijo = calcular_incumplimiento(hijos[i], problema_id)

        if inc_hijo == 0 and inc_act == 0:
            nueva_poblacion[i] = hijos[i] if obj_hijo < obj_act else poblacion[i]
        elif inc_hijo == 0:
            nueva_poblacion[i] = hijos[i]
        elif inc_act == 0:
            nueva_poblacion[i] = poblacion[i]
        else:
            nueva_poblacion[i] = hijos[i] if inc_hijo < inc_act else poblacion[i]
    return nueva_poblacion

def resolver_suma_ponderada(pesos, problema_id, D, lim_inf, lim_sup, NP, F, CR, GEN):
    poblacion = crear_poblacion_inicial(D, lim_inf, lim_sup, NP)
    for _ in range(GEN):
        mutados = mutar(poblacion, F, lim_inf, lim_sup)
        cruzados = cruzar(poblacion, mutados, CR)
        poblacion = seleccionar(poblacion, cruzados, pesos, problema_id)

    mejor = min(
        poblacion,
        key=lambda x: (calcular_incumplimiento(x, problema_id), calcular_objetivo(x, pesos, problema_id))
    )
    return mejor, funciones(mejor, problema_id)

# =======================
# INTERFAZ STREAMLIT
# =======================

def mostrar():
    st.set_page_config(layout="wide")
    st.title("Método de Suma Ponderada")

    col1, col2 = st.columns(2)

    with st.sidebar:
        st.header("Configuración")
        problema_id = st.selectbox("Selecciona el problema", [0, 1, 2, 3])
        num_pesos = st.slider("Número de puntos en el frente", 5, 100, 20)

        # Parámetros modificables
        NP = st.number_input("Tamaño de población (NP)", min_value=10, max_value=500, value=50, step=10)
        F = st.slider("Factor de mutación (F)", 0.1, 1.0, 0.45, 0.05)
        CR = st.slider("Probabilidad de cruce (CR)", 0.0, 1.0, 0.8, 0.05)
        GEN = st.number_input("Número de generaciones (GEN)", min_value=10, max_value=1000, value=100, step=10)

    D, lim_inf, lim_sup = obtener_parametros(problema_id)
    pesos_list = np.linspace(0, 1, num_pesos)
    pareto = []

    for w1 in pesos_list:
        w2 = 1.0 - w1
        pesos = [w1, w2]
        x, objetivos = resolver_suma_ponderada(pesos, problema_id, D, lim_inf, lim_sup, NP, F, CR, GEN)
        if calcular_incumplimiento(x, problema_id) == 0:
            pareto.append(objetivos)

    pareto = np.array(pareto)

    st.subheader("Frente de Pareto generado")
    fig, ax = plt.subplots()
    if len(pareto) > 0:
        indices = np.argsort(pareto[:, 0])
        ax.scatter(pareto[:, 0], pareto[:, 1], c='purple', s=20, label=f"Soluciones = {len(pareto)}")
        ax.plot(pareto[indices, 0], pareto[indices, 1], linestyle='--', linewidth=1, color='magenta')
    ax.set_xlabel("f1(x)")
    ax.set_ylabel("f2(x)")
    ax.set_title(f"Frente de Pareto - Problema {problema_id}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    with col1:
        st.subheader("Método de Suma Ponderada")
        st.markdown("### **¿Qué es el Método de Suma Ponderada?**")
        st.markdown("""
        Es una técnica que convierte un problema con **múltiples objetivos** en uno con **un solo objetivo**, 
        combinando todas las funciones objetivo usando pesos específicos.
        """)
        st.markdown("### **Paso 1: Problema Original (Multiobjetivo)**")
        st.latex(r"\text{Minimizar} \quad \mathbf{F}(x) = [F_1(x), F_2(x), \dots, F_k(x)]^T")
        st.latex(r"\text{sujeto a:} \quad g_j(x) \leq 0, \quad j = 1, 2, \dots, m")
        st.latex(r"x \in \mathbb{R}^n")
        st.markdown("### **Paso 2: Transformación con Pesos**")
        st.latex(r"U(x) = \sum_{i=1}^{k} w_i F_i(x)")
        st.latex(r"w_i \geq 0 \quad \text{y} \quad \sum_{i=1}^{k} w_i = 1")
        st.markdown("### **Paso 3: Problema Escalarizado**")
        st.latex(r"\text{Minimizar} \quad U(x)")
        st.latex(r"\text{sujeto a:} \quad g_j(x) \leq 0")
        st.markdown("### **Paso 4: Frente de Pareto**")
        st.markdown("Se resuelven múltiples problemas variando los pesos para generar el frente.")

    with col2:
        st.subheader("Algoritmo Paso a Paso")

        st.markdown("### **1. Generar conjunto de pesos**")
        st.code('''
pesos_list = np.linspace(0, 1, num_pesos)
pareto = []
        ''', language="python")

        st.markdown("### **2. Resolver para cada combinación**")
        st.code('''
for w1 in pesos_list:
    w2 = 1.0 - w1
    pesos = [w1, w2]
    x, objetivos = resolver_suma_ponderada(pesos)
    if calcular_incumplimiento(x) == 0:
        pareto.append(objetivos)
        ''', language="python")

        st.markdown("### **3. Evaluar función objetivo escalarizada**")
        st.code('''
def calcular_objetivo(x, pesos):
    f1, f2 = funciones(x)
    return pesos[0] * f1 + pesos[1] * f2
        ''', language="python")

        st.markdown("### **4. Optimización con DE y Reglas de Deb**")
        st.code('''
def resolver_suma_ponderada(pesos):
    poblacion = crear_poblacion_inicial()
    for _ in range(GEN):
        mutados = mutar(poblacion, F)
        cruzados = cruzar(poblacion, mutados, CR)
        poblacion = seleccionar(poblacion, cruzados, pesos)
    return mejor_solucion(poblacion)
        ''', language="python")

        st.markdown("### **5. Frente de Pareto final**")
        st.code('''
pareto = np.array(pareto)
pareto = pareto[pareto[:, 0].argsort()]
        ''', language="python")
        st.success("El conjunto de soluciones factibles construye el frente de Pareto.")
