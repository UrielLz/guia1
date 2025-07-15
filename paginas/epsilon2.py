import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from funciones2 import funciones, restricciones, obtener_parametros

np.random.seed(15)

# ==========================
# FUNCIONES AUXILIARES
# ==========================

def calcular_incumplimiento(x, problema_id, epsilon, idx_rest):
    f1, f2 = funciones(x, problema_id)
    obj_rest = [f1, f2]
    r = restricciones(x, problema_id)
    total = 0

    # Restricciones originales del problema
    for val in r:
        if val > 0:
            total += val

    # Restricción epsilon (f_j <= ε)
    if obj_rest[idx_rest] > epsilon:
        total += obj_rest[idx_rest] - epsilon

    return total

def calcular_objetivo(x, idx_obj, problema_id):
    f1, f2 = funciones(x, problema_id)
    return [f1, f2][idx_obj]

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

def seleccionar(poblacion, hijos, idx_obj, problema_id, epsilon, idx_rest):
    NP = poblacion.shape[0]
    nueva_poblacion = np.zeros_like(poblacion)
    for i in range(NP):
        obj1 = calcular_objetivo(poblacion[i], idx_obj, problema_id)
        obj2 = calcular_objetivo(hijos[i], idx_obj, problema_id)
        inc1 = calcular_incumplimiento(poblacion[i], problema_id, epsilon, idx_rest)
        inc2 = calcular_incumplimiento(hijos[i], problema_id, epsilon, idx_rest)

        if inc1 == 0 and inc2 == 0:
            nueva_poblacion[i] = hijos[i] if obj2 < obj1 else poblacion[i]
        elif inc2 == 0:
            nueva_poblacion[i] = hijos[i]
        elif inc1 == 0:
            nueva_poblacion[i] = poblacion[i]
        else:
            nueva_poblacion[i] = hijos[i] if inc2 < inc1 else poblacion[i]
    return nueva_poblacion

def resolver_epsilon(idx_obj, idx_rest, epsilon, problema_id, D, lim_inf, lim_sup, NP, F, CR, GEN):
    poblacion = crear_poblacion_inicial(D, lim_inf, lim_sup, NP)
    for _ in range(GEN):
        mutados = mutar(poblacion, F, lim_inf, lim_sup)
        cruzados = cruzar(poblacion, mutados, CR)
        poblacion = seleccionar(poblacion, cruzados, idx_obj, problema_id, epsilon, idx_rest)

    factibles = [
        (x, calcular_objetivo(x, idx_obj, problema_id))
        for x in poblacion
        if calcular_incumplimiento(x, problema_id, epsilon, idx_rest) == 0
    ]
    if not factibles:
        return None, None
    x_best, _ = min(factibles, key=lambda item: item[1])
    return x_best, funciones(x_best, problema_id)

# ==========================
# STREAMLIT UI
# ==========================

def mostrar():
    st.set_page_config(layout="wide")
    st.title("Método de Restricciones ε (Epsilon-Constraint)")

    col1, col2 = st.columns(2)

    with st.sidebar:
        st.header("Configuración")
        problema_id = st.selectbox("Selecciona el problema", [0, 1, 2, 3])
        idx_obj = st.selectbox("Función a minimizar", [0, 1], format_func=lambda i: f"f{i+1}")
        num_eps = st.slider("Número de valores ε", 5, 50, 20)

        NP = st.number_input("Tamaño de población (NP)", min_value=10, max_value=500, value=50, step=10)
        F = st.slider("Factor de mutación (F)", 0.1, 1.0, 0.45, 0.05)
        CR = st.slider("Probabilidad de cruce (CR)", 0.0, 1.0, 0.8, 0.05)
        GEN = st.number_input("Número de generaciones (GEN)", min_value=10, max_value=1000, value=100, step=10)

    D, lim_inf, lim_sup = obtener_parametros(problema_id)
    idx_rest = 1 - idx_obj  # f2 si f1 es objetivo, y viceversa

    # Calcular rango de epsilon
    muestras = [funciones(np.random.uniform(lim_inf, lim_sup), problema_id) for _ in range(500)]
    f_rest_values = np.array([m[idx_rest] for m in muestras])
    eps_min, eps_max = np.min(f_rest_values), np.max(f_rest_values)
    epsilon_vals = np.linspace(eps_min, eps_max, num_eps)

    pareto = []
    for epsilon in epsilon_vals:
        x, objetivos = resolver_epsilon(idx_obj, idx_rest, epsilon, problema_id, D, lim_inf, lim_sup, NP, F, CR, GEN)
        if x is not None:
            pareto.append(objetivos)

    pareto = np.array(pareto)

    st.subheader("Frente de Pareto generado")
    fig, ax = plt.subplots()
    if len(pareto) > 0:
        indices = np.argsort(pareto[:, 0])
        ax.scatter(pareto[:, 0], pareto[:, 1], c='green', s=20, label=f"Soluciones = {len(pareto)}")
        ax.plot(pareto[indices, 0], pareto[indices, 1], linestyle='--', linewidth=1, color='darkgreen')
    ax.set_xlabel("f1(x)")
    ax.set_ylabel("f2(x)")
    ax.set_title(f"Frente de Pareto - Problema {problema_id}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    with col1:
        st.subheader("Método de Restricciones ε (Epsilon-Constraint)")

        st.markdown("### **¿Qué es el Método ε-Constraint?**")
        st.markdown("""
        Es un enfoque que convierte un problema **multiobjetivo** en uno **monoobjetivo**. 
        Se escoge una de las funciones objetivo para **minimizar**, y la(s) otra(s) se convierten en **restricciones con un umbral ε**.
        """)

        st.markdown("### **Paso 1: Problema Multiobjetivo Original**")
        st.latex(r"\text{Minimizar} \quad \mathbf{F}(x) = [F_1(x), F_2(x), \dots, F_k(x)]^T")
        st.latex(r"\text{sujeto a:} \quad g_j(x) \leq 0, \quad j = 1, 2, \dots, m")
        st.latex(r"x \in \mathbb{R}^n")

        st.markdown("### **Paso 2: Convertir a Problema Monoobjetivo**")
        st.markdown("Seleccionamos una función objetivo principal $F_i(x)$ a minimizar, y las demás se convierten en restricciones:")
        st.latex(r"\text{Minimizar} \quad F_i(x)")
        st.latex(r"\text{sujeto a:} \quad F_j(x) \leq \epsilon \quad (j \neq i)")
        st.latex(r"\quad \text{y} \quad g_j(x) \leq 0")

        st.markdown("### **Paso 3: Generar Frente de Pareto**")
        st.markdown("""
        Se generan múltiples problemas monoobjetivo variando ε en un rango. Cada solución factible obtenida 
        representa un punto diferente en el **frente de Pareto**.
        """)

        st.info("Este método permite explorar el frente de Pareto sin necesidad de combinar funciones objetivo.")

    with col2:
        st.subheader("Algoritmo Paso a Paso")

        st.markdown("### **1. Seleccionar función a optimizar y función restringida**")
        st.markdown("""
Del conjunto de funciones objetivo $F(x) = [f_1(x), f_2(x)]$, se selecciona una para **minimizar**
y la otra se transforma en una **restricción adicional** del tipo $f_j(x) \leq \epsilon$.
        """)

        st.code('''
# Selección de índices
idx_obj = 0      # f1 será la función objetivo
idx_rest = 1     # f2 se usará como restricción ε
        ''', language="python")

        st.markdown("### **2. Definir el conjunto de valores ε**")
        st.markdown("""
Se evalúa $f_j(x)$ sobre muestras aleatorias para obtener el rango adecuado de valores ε.
Luego, se discretiza ese rango para construir diferentes subproblemas escalarizados.
        """)

        st.code('''
# Evaluar f_j(x) en muestras aleatorias
muestras = [funciones(np.random.uniform(lim_inf, lim_sup), problema_id) for _ in range(500)]
valores_fj = np.array([m[idx_rest] for m in muestras])

# Definir límites y valores de ε
eps_min, eps_max = np.min(valores_fj), np.max(valores_fj)
epsilons = np.linspace(eps_min, eps_max, num_eps)
        ''', language="python")

        st.markdown("### **3. Resolver cada subproblema con DE + Reglas de Deb**")
        st.markdown("""
Para cada ε, se aplica un proceso de **Evolución Diferencial (DE)** que minimiza $f_i(x)$ 
sujeto a las restricciones originales y a la nueva condición $f_j(x) \leq \epsilon$.
        """)

        st.code('''
pareto = []
for epsilon in epsilons:
    x, obj = resolver_epsilon(idx_obj, idx_rest, epsilon, ...)
    if x is not None:
        pareto.append(obj)
        ''', language="python")

        st.markdown("### **4. Evaluar la factibilidad de cada solución**")
        st.markdown("""
Se utiliza la función de **incumplimiento total**, que suma todas las violaciones tanto
de las restricciones del problema como de la restricción ε impuesta a $f_j(x)$.
        """)

        st.code('''
def calcular_incumplimiento(x, problema_id, epsilon, idx_rest):
    r = restricciones(x, problema_id)
    f_rest = funciones(x, problema_id)[idx_rest]
    total = sum(max(0, val) for val in r)
    if f_rest > epsilon:
        total += f_rest - epsilon
    return total
        ''', language="python")

        st.markdown("### **5. Construir el Frente de Pareto**")
        st.markdown("""
Las soluciones factibles se ordenan y visualizan, generando un conjunto representativo del 
**frente de Pareto** bajo el esquema ε-constraint.
        """)

        st.code('''
pareto = np.array(pareto)
pareto = pareto[pareto[:, 0].argsort()]  # ordenar por f1
        ''', language="python")

        st.success("El método ε-Constraint construye el frente de Pareto resolviendo múltiples subproblemas escalarizados.")
