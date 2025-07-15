import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from funciones2 import funciones, restricciones, obtener_parametros

np.random.seed(15)

def mostrar():
    st.set_page_config(layout="wide")
    st.title("Obtención de Metas")

    with st.sidebar:
        st.header("Configuración")
        problema_id = st.selectbox("Selecciona el problema", [0, 1, 2, 3])
        NP = st.slider("Tamaño de población", 10, 200, 30, step=10)
        F = st.slider("Factor de mutación (F)", 0.1, 1.0, 0.7, step=0.05)
        CR = st.slider("Probabilidad de cruce (CR)", 0.0, 1.0, 0.9, step=0.05)
        GEN = st.slider("Número de generaciones", 10, 1000, 100, step=10)
        num_pesos = st.slider("Número de puntos", 10, 100, 20, step=10)

    metas = {
        0: [0.0, 4.0],
        1: [0, 0],
        2: [-3, -5],
        3: [-130.0, -250.0]
    }[problema_id]

    D_x, lim_inf_x, lim_sup_x = obtener_parametros(problema_id)
    D = D_x + 1
    lim_inf = np.append(lim_inf_x, -500)
    lim_sup = np.append(lim_sup_x, 500)

    def crear_poblacion_inicial():
        return np.random.uniform(lim_inf, lim_sup, (NP, D))

    def aplicar_limites(poblacion):
        return np.clip(poblacion, lim_inf, lim_sup)

    def mutar(poblacion):
        nuevos = np.zeros_like(poblacion)
        for i in range(NP):
            idx = list(range(NP))
            idx.remove(i)
            r1, r2, r3 = np.random.choice(idx, 3, replace=False)
            nuevos[i] = poblacion[r3] + F * (poblacion[r1] - poblacion[r2])
        return aplicar_limites(nuevos)

    def cruzar(poblacion, mutados):
        hijos = np.zeros_like(poblacion)
        for i in range(NP):
            punto = np.random.randint(0, D)
            for j in range(D):
                hijos[i, j] = mutados[i, j] if np.random.rand() < CR or j == punto else poblacion[i, j]
        return hijos

    def calcular_objetivo(ind):
        return ind[-1]

    def calcular_incumplimiento(ind, metas_f, w_k):
        x = ind[:-1]
        lam = ind[-1]
        inc = sum(max(0, r) for r in restricciones(x, problema_id))
        f_vals = funciones(x, problema_id)
        for k in range(len(metas_f)):
            inc += max(0, f_vals[k] - w_k[k] * lam - metas_f[k])
        return inc

    def seleccionar(poblacion, hijos, metas_f, w_k):
        nueva = np.zeros_like(poblacion)
        for i in range(NP):
            obj1 = calcular_objetivo(poblacion[i])
            inc1 = calcular_incumplimiento(poblacion[i], metas_f, w_k)
            obj2 = calcular_objetivo(hijos[i])
            inc2 = calcular_incumplimiento(hijos[i], metas_f, w_k)
            if inc1 == 0 and inc2 == 0:
                nueva[i] = hijos[i] if obj2 < obj1 else poblacion[i]
            elif inc2 == 0:
                nueva[i] = hijos[i]
            elif inc1 == 0:
                nueva[i] = poblacion[i]
            else:
                nueva[i] = hijos[i] if inc2 < inc1 else poblacion[i]
        return nueva

    def resolver(metas_f, w_k):
        poblacion = crear_poblacion_inicial()
        for _ in range(GEN):
            mutados = mutar(poblacion)
            cruzados = cruzar(poblacion, mutados)
            poblacion = seleccionar(poblacion, cruzados, metas_f, w_k)
        mejor = min(poblacion, key=lambda ind: (
            calcular_incumplimiento(ind, metas_f, w_k),
            calcular_objetivo(ind)
        ))
        x_opt = mejor[:-1]
        f_vals = funciones(x_opt, problema_id)
        inc_gam = calcular_incumplimiento(mejor, metas_f, w_k)
        inc_orig = sum(max(0, r) for r in restricciones(x_opt, problema_id))
        return x_opt, f_vals, inc_gam, inc_orig

    pesos_list = np.linspace(0, 1, num_pesos)
    pareto = []

    for i, w1 in enumerate(pesos_list):
        w2 = 1.0 - w1
        w_k = np.array([w1, w2])
        if np.sum(w_k) < 1e-9:
            w_k = np.array([1e-6, 1.0 - 1e-6])
        if w_k[1] < 0:
            w_k[1] = 1e-6
        x_sol, f_sol, inc_gam, inc_orig = resolver(metas, w_k)
        if inc_orig < 1e-5 and inc_gam < 1e-4:
            pareto.append(f_sol)

    pareto = np.array(pareto)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("¿Qué es la Obtención de Metas?")
        
        st.markdown("""
    La **Obtención de Metas** es una técnica que permite encontrar una solución equilibrada en un problema multiobjetivo
    mediante la minimización de una variable auxiliar λ. Esta variable representa el grado de satisfacción de las metas deseadas.
    """)

        st.markdown("""
    Para cada función objetivo $f_k(x)$, se define una meta deseada $f_{d,k}$.  
    El cumplimiento de esta meta se expresa mediante la siguiente **restricción de metas ajustada**:
    """)

        st.latex(r"f_k(x) \leq w_k \cdot \lambda + f_{d,k} \quad \text{para } k = 1, \ldots, K")

        st.markdown("""
    **Donde:**
    - $w_k$: peso asignado a la función $f_k(x)$
    - $f_{d,k}$: valor deseado para $f_k(x)$
    """)

        st.markdown("""
    Se busca minimizar $\\lambda$ sujeto a:
    - Las **restricciones originales** del problema
    - Las **restricciones de metas ajustadas** usando $\\lambda$
    """)

        st.markdown("**El modelo completo se define como:**")

        st.latex(r"""
    \begin{aligned}
    \text{Minimizar} \quad & \lambda \\
    \text{sujeto a} \quad & g_j(x) \leq 0, \quad j = 1, \ldots, m \\
    & f_k(x) \leq w_k \cdot \lambda + f_{d,k}, \quad k = 1, \ldots, K
    \end{aligned}
    """)

        st.markdown("""
    La minimización de $\\lambda$ garantiza que las funciones objetivo se mantengan tan cercanas como sea posible
    a los valores deseados $f_{d,k}$, ponderadas por su importancia relativa $w_k$.
    """)
    with col2:
        st.subheader("Algoritmo Paso a Paso")

        st.markdown("### 1. Ampliar el vector de decisión con la variable auxiliar $\\lambda$")
        st.markdown("""
Se añade una variable adicional al vector de decisión:  
- Las primeras $D$ variables corresponden a las decisiones originales $x$.  
- La última variable es $\\lambda$, que se va a minimizar.
        """)
        st.code("dimension = D + 1  # variables originales + lambda")

        st.markdown("### 2. Definir los límites de búsqueda para cada variable")
        st.markdown("""
Se ajustan los límites inferiores y superiores para incluir también $\\lambda$:
- Las variables $x_i$ conservan sus límites originales.  
- $\\lambda$ se acota en $[-500, 500]$ para mantener un rango razonable de búsqueda.
        """)
        st.code("lim_inf = np.append(lim_inf_x, -500)\nlim_sup = np.append(lim_sup_x, 500)")

        st.markdown("### 3. Inicializar población aleatoria en el nuevo espacio de búsqueda")
        st.markdown("""
Se genera una población inicial de tamaño $NP$ con valores uniformemente distribuidos entre los nuevos límites.
        """)
        st.code("poblacion = crear_poblacion_inicial()")

        st.markdown("### 4. Optimizar con Evolución Diferencial (DE) + Reglas de Deb")
        st.markdown("""
Se aplica el algoritmo DE para evolucionar la población:
- **Mutación**: se crean individuos perturbados.  
- **Cruce**: se recombinan genes entre individuos.  
- **Selección**: se escoge el mejor individuo considerando:
  - El **cumplimiento de restricciones originales** $g_j(x) \\leq 0$
  - El **cumplimiento de metas ajustadas** $f_k(x) \\leq w_k \\cdot \\lambda + f_{d,k}$
  - El **valor más bajo de $\\lambda$**
        """)
        st.code('''for _ in range(GEN):
    mutados = mutar(poblacion)
    cruzados = cruzar(poblacion, mutados)
    poblacion = seleccionar(poblacion, cruzados, metas, pesos_w)
''')

        st.markdown("### 5. Evaluar factibilidad para cada individuo")
        st.markdown("""
Para cada individuo se evalúa si cumple las condiciones:
- $g_j(x) \\leq 0$  (restricciones originales)  
- $f_k(x) \\leq w_k \\cdot \\lambda + f_{d,k}$ (restricciones de metas ajustadas)

        """)
        st.latex(r"f_k(x) \leq w_k \cdot \lambda + f_{d,k}, \quad \forall k")
        st.latex(r"g_j(x) \leq 0, \quad \forall j")

        st.markdown("### 6. Seleccionar el mejor individuo factible")
        st.markdown("""
Entre todos los individuos que cumplen las restricciones, se selecciona el que tenga el menor valor de $\\lambda$.
        """)
        st.code('''mejor = min(poblacion, key=lambda ind: (
    calcular_incumplimiento(ind, metas, pesos_w),
    calcular_objetivo(ind)
))''')
    st.subheader("Frente obtenido para obtención de metas")
    fig, ax = plt.subplots()
    if len(pareto) > 0:
        idx = np.argsort(pareto[:, 0])
        ax.scatter(pareto[:, 0], pareto[:, 1], c='purple', s=20, label=f"Soluciones = {len(pareto)}")
        ax.plot(pareto[idx, 0], pareto[idx, 1], linestyle='--', linewidth=1, color='magenta')
    ax.set_xlabel("f1(x)")
    ax.set_ylabel("f2(x)")
    ax.set_title(f"Frente de Pareto - Obtención de Metas (Problema {problema_id})")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
