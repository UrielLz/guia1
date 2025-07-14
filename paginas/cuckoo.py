import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import random
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
        <h1 style='text-align: center; color: white; background-color: #1E1E1E; padding: 1rem; border-radius: 0.5rem;'>Cuckoo Search (CS)</h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    st.sidebar.title("Configuración del problema")
    
    # Bloques explicativos del algoritmo
    bloques = {
        "Inicialización": {
            "exp": r"""
**Fórmula:**
$
x_{i,j} = x_j^L + rand() \cdot (x_j^U - x_j^L)
$
Se genera una población inicial de $NP$ nidos (soluciones candidatas) distribuidos aleatoriamente dentro del espacio de búsqueda definido por los límites del problema.

**Variables:**
- $x_{i,j}$: Valor de la j-ésima variable del i-ésimo nido
- $x_j^L$, $x_j^U$: Límite inferior y superior de la variable j
- $rand()$: Número aleatorio uniforme ∈ [0,1]
- $NP$: Tamaño de la población
- $D$: Dimensión del problema
""",
            "code": "# Inicialización de la población\nnidos = lim_inf + np.random.rand(NP, D) * (lim_sup - lim_inf)\n\n# Encontrar la mejor solución inicial\nmejor_nido = nidos[0].copy()\nmejor_score = funcion_objetivo(mejor_nido, pid)\nfor i in range(1, NP):\n    if es_mejor_cs(nidos[i], mejor_nido):\n        mejor_nido = nidos[i].copy()\n        mejor_score = funcion_objetivo(mejor_nido, pid)"
        },
        "Evaluación y Restricciones": {
            "exp": r"""
**Fórmula de penalización:**
$
\phi(x) = \sum_{j=1}^{m} \max(0, g_j(x))
$

**Reglas de Deb para comparación:**
1. Si ambas soluciones son factibles → comparar función objetivo
2. Si una es factible y otra no → preferir la factible
3. Si ambas son no factibles → preferir la de menor violación

**Variables:**
- $g_j(x)$: j-ésima restricción de desigualdad
- $\phi(x)$: Suma total de violaciones
- $TOL$: Tolerancia para considerar una solución factible
""",
            "code": "def calcular_incumplimiento(x):\n    return np.sum(np.maximum(0, restricciones(x, pid)))\n\ndef es_mejor_cs(x1, x2):\n    f1 = funcion_objetivo(x1, pid)\n    f2 = funcion_objetivo(x2, pid)\n    incum1 = calcular_incumplimiento(x1)\n    incum2 = calcular_incumplimiento(x2)\n    \n    if incum1 <= TOL and incum2 <= TOL:\n        return f1 < f2  # Ambas factibles\n    elif incum1 <= TOL and incum2 > TOL:\n        return True     # x1 factible, x2 no\n    elif incum1 > TOL and incum2 <= TOL:\n        return False    # x1 no factible, x2 sí\n    else:\n        return incum1 < incum2  # Ambas no factibles"
        },
        "Vuelo Lévy": {
            "exp": r"""
**Distribución de Lévy:**
$
\text{paso} = \alpha \cdot \frac{u \cdot \sigma}{|v|^{1/\lambda}}
$

**Parámetro sigma:**
$
\sigma = \left(\frac{\Gamma(1+\lambda) \sin(\pi\lambda/2)}{\Gamma((1+\lambda)/2) \lambda 2^{(\lambda-1)/2}}\right)^{1/\lambda}
$

Los vuelos Lévy generan movimientos con pasos largos ocasionales y muchos pasos cortos, imitando el comportamiento de búsqueda de los cucos.

**Variables:**
- $u$, $v$: Variables aleatorias normales estándar
- $\lambda$: Exponente de Lévy (típicamente 1.5)
- $\alpha$: Tamaño del paso (se reduce con el tiempo)
- $\Gamma$: Función gamma
""",
            "code": "def paso_levy(dim, alpha_paso_actual):\n    u = np.random.normal(0, 1, dim)\n    v = np.random.normal(0, 1, dim)\n    \n    # Cálculo del parámetro sigma\n    sigma_u = (math.gamma(1 + LAMBDA) * np.sin(np.pi * LAMBDA / 2) /\n              (math.gamma((1 + LAMBDA) / 2) * LAMBDA * 2 ** ((LAMBDA - 1) / 2))) ** (1 / LAMBDA)\n    \n    levy_step = alpha_paso_actual * u * sigma_u / np.abs(v) ** (1 / LAMBDA)\n    return levy_step\n\n# Aplicación del vuelo Lévy\nfor i in range(NP):\n    paso = paso_levy(D, alpha_actual)\n    candidato = nidos[i] + paso\n    candidato = np.clip(candidato, lim_inf, lim_sup)"
        },
        "Abandono de Nidos": {
            "exp": r"""
**Probabilidad de abandono:**
$
P_a = 0.25 \text{ (típicamente)}
$

**Procedimiento:**
1. Se calcula un score penalizado para cada nido
2. Se identifican los $\lfloor P_a \times NP \rfloor$ peores nidos
3. Se reemplazan por nuevos nidos aleatorios si mejoran

**Score penalizado:**
$
score_i = f(x_i) + 10^6 \times \max(0, \phi(x_i))
$

Este mecanismo simula el descubrimiento de huevos por parte de los pájaros huésped.
""",
            "code": "# Calcular número de nidos a abandonar\nn_abandonar = int(Pa * NP)\n\nif n_abandonar > 0:\n    # Calcular scores penalizados\n    fitness_scores = []\n    for i in range(NP):\n        f_val = funcion_objetivo(nidos[i], pid)\n        incum = calcular_incumplimiento(nidos[i])\n        score = f_val + 1e6 * max(0, incum)\n        fitness_scores.append(score)\n    \n    # Encontrar los peores nidos\n    peores_indices = np.argsort(fitness_scores)[-n_abandonar:]\n    \n    # Reemplazar si mejoran\n    for idx in peores_indices:\n        nuevo = lim_inf + np.random.rand(D) * (lim_sup - lim_inf)\n        if es_mejor_cs(nuevo, nidos[idx]):\n            nidos[idx] = nuevo"
        },
        "Actualización y Convergencia": {
            "exp": r"""
**Reducción adaptativa del paso:**
$
\alpha_t = \alpha_0 \times \rho^t
$

**Criterio de actualización:**
- Se mantiene siempre la mejor solución encontrada
- Solo se actualiza si se encuentra una solución mejor
- El algoritmo converge cuando no hay mejoras significativas

**Variables:**
- $\alpha_0$: Tamaño inicial del paso
- $\rho$: Factor de reducción (0 < ρ < 1)
- $t$: Número de generación actual
""",
            "code": "# Reducción adaptativa del paso\nalpha_actual = ALPHA_INICIAL * (REDUCTOR ** g)\n\n# Actualización de la mejor solución\nfor i in range(NP):\n    if es_mejor_cs(nidos[i], mejor_nido):\n        mejor_nido = nidos[i].copy()\n        mejor_score = funcion_objetivo(mejor_nido, pid)\n\n# Guardar progreso\nhistorial.append(mejor_score)\n\n# Condición de terminación\nif g >= generaciones:\n    break"
        }
    }

    problemas_benchmark = {f"g{str(i+1).zfill(2)}": i for i in range(24)}
    opcion_problema = st.sidebar.selectbox("Selecciona un problema:", list(problemas_benchmark.keys()))
    pid = problemas_benchmark[opcion_problema]
    D, lim_inf, lim_sup = obtener_parametros(pid)

    NP = st.sidebar.slider("Tamaño de población (NP)", 10, 100, 30, 5)
    generaciones = st.sidebar.slider("Generaciones", 10, 200, 50, step=10)
    Pa = st.sidebar.slider("Probabilidad de abandono (Pa)", 0.0, 1.0, 0.25, 0.05)
    TOL = st.sidebar.number_input("Tolerancia", value=1e-6, format="%e")
    mostrar_optimo = st.sidebar.checkbox("Marcar mejor solución", value=True)

    ALPHA_INICIAL = 1.0
    REDUCTOR = 0.95  # Reducción más suave
    LAMBDA = 1.5

    def paso_levy(dim, alpha_paso_actual):
        """Genera un paso siguiendo distribución de Lévy"""
        u = np.random.normal(0, 1, dim)
        v = np.random.normal(0, 1, dim)
        
        # Cálculo del parámetro sigma para la distribución de Lévy
        sigma_u = (math.gamma(1 + LAMBDA) * np.sin(np.pi * LAMBDA / 2) /
                  (math.gamma((1 + LAMBDA) / 2) * LAMBDA * 2 ** ((LAMBDA - 1) / 2))) ** (1 / LAMBDA)
        
        levy_step = alpha_paso_actual * u * sigma_u / np.abs(v) ** (1 / LAMBDA)
        return levy_step

    def calcular_incumplimiento(x):
        """Calcula la suma de violaciones de restricciones"""
        return np.sum(np.maximum(0, restricciones(x, pid)))

    def es_mejor_cs(x1, x2):
        """Compara dos soluciones usando manejo de restricciones"""
        f1 = funcion_objetivo(x1, pid)
        f2 = funcion_objetivo(x2, pid)
        incum1 = calcular_incumplimiento(x1)
        incum2 = calcular_incumplimiento(x2)

        # Reglas de Deb para manejo de restricciones
        if incum1 <= TOL and incum2 <= TOL:
            # Ambas son factibles, comparar función objetivo
            return f1 < f2
        elif incum1 <= TOL and incum2 > TOL:
            # x1 es factible, x2 no
            return True
        elif incum1 > TOL and incum2 <= TOL:
            # x1 no es factible, x2 sí
            return False
        else:
            # Ambas son no factibles, comparar violaciones
            return incum1 < incum2

    def ejecutar_cs():
        # Inicialización de la población
        nidos = lim_inf + np.random.rand(NP, D) * (lim_sup - lim_inf)
        historial = []

        # Encontrar la mejor solución inicial
        mejor_nido = nidos[0].copy()
        mejor_score = funcion_objetivo(mejor_nido, pid)
        
        for i in range(1, NP):
            if es_mejor_cs(nidos[i], mejor_nido):
                mejor_nido = nidos[i].copy()
                mejor_score = funcion_objetivo(mejor_nido, pid)

        for g in range(generaciones):
            alpha_actual = ALPHA_INICIAL * (REDUCTOR ** g)
            nuevos_nidos = nidos.copy()

            # Fase 1: Generación de nuevos candidatos por vuelo Lévy
            for i in range(NP):
                # Generar nuevo candidato con vuelo Lévy
                paso = paso_levy(D, alpha_actual)
                candidato = nidos[i] + paso
                candidato = np.clip(candidato, lim_inf, lim_sup)
                
                # Comparar con un nido aleatorio
                j = random.randint(0, NP - 1)
                while j == i:
                    j = random.randint(0, NP - 1)
                
                if es_mejor_cs(candidato, nidos[j]):
                    nuevos_nidos[j] = candidato

            nidos = nuevos_nidos

            # Fase 2: Abandono de nidos con probabilidad Pa
            n_abandonar = int(Pa * NP)
            if n_abandonar > 0:
                # Calcular fitness para ordenar
                fitness_scores = []
                for i in range(NP):
                    f_val = funcion_objetivo(nidos[i], pid)
                    incum = calcular_incumplimiento(nidos[i])
                    # Penalizar soluciones no factibles
                    score = f_val + 1e6 * max(0, incum)
                    fitness_scores.append(score)
                
                # Encontrar los peores nidos
                peores_indices = np.argsort(fitness_scores)[-n_abandonar:]
                
                for idx in peores_indices:
                    # Generar nuevo nido aleatorio
                    nuevo = lim_inf + np.random.rand(D) * (lim_sup - lim_inf)
                    if es_mejor_cs(nuevo, nidos[idx]):
                        nidos[idx] = nuevo

            # Actualizar la mejor solución
            for i in range(NP):
                if es_mejor_cs(nidos[i], mejor_nido):
                    mejor_nido = nidos[i].copy()
                    mejor_score = funcion_objetivo(mejor_nido, pid)

            # Guardar el progreso
            historial.append(mejor_score)

        return historial, mejor_nido, mejor_score

    # Ejecutar el algoritmo
    historial, mejor_sol, mejor_val = ejecutar_cs()

    # Interfaz explicativa
    with col1:
        st.subheader("Pseudocódigo")
        paso = st.radio("Selecciona un bloque:", list(bloques.keys()), horizontal=True)
        st.markdown(bloques[paso]['exp'], unsafe_allow_html=True)
    
    with col2:
        st.subheader(f"Código del paso: {paso}")
        st.code(bloques[paso]['code'], language="python")

    st.subheader("Evolución de la solución")
    fig, ax = plt.subplots()
    valores_limpios = [v for v in historial if v is not None]
    ax.plot(valores_limpios, color=PALETA["linea"], label="f(x) mínimo")
    if mostrar_optimo and valores_limpios:
        idx = np.argmin(valores_limpios)
        ax.plot(idx, valores_limpios[idx], 'o', color=PALETA["mejor_punto"], label="Mejor f(x)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("f(x)")
    ax.set_title("Optimización con Cuckoo Search")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    if mejor_sol is not None:
        st.success(f"Mejor solución encontrada: x = {np.round(mejor_sol, 4).tolist()}, f(x) = {mejor_val:.6f}")