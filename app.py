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

st.set_page_config(layout="wide")
st.markdown("""
<div style='background-color:#1E1E1E;padding:20px;border-radius:10px'>
    <h1 style='color:white;text-align:center;'>Evolución Diferencial (DE) con Deb</h1>
</div>
""", unsafe_allow_html=True)


# Layout
col1, col2 = st.columns(2)


st.sidebar.title(" Configuración del problema ")

problemas_benchmark = {
    "g01": 0, "g02": 1, "g03": 2, "g04": 3, "g05": 4, "g06": 5, "g07": 6,
    "g08": 7, "g09": 8, "g10": 9, "g11": 10, "g12": 11, "g13": 12, "g14": 13,
    "g15": 14, "g16": 15, "g17": 16, "g18": 17, "g19": 18, "g20": 19, "g21": 20,
    "g22": 21, "g23": 22, "g24": 23
}

opcion_problema = st.sidebar.selectbox("Selecciona un problema :", list(problemas_benchmark.keys()))
pid = problemas_benchmark[opcion_problema]

D, lim_inf_vec, lim_sup_vec = obtener_parametros(pid)

NP = st.sidebar.slider("Tamaño de población (NP)", 10, 100, 30, 5)
F = st.sidebar.slider("Factor de mutación (F)", 0.0, 1.0, 0.5)
CR = st.sidebar.slider("Tasa de cruce (CR)", 0.0, 1.0, 0.9)
generaciones = st.sidebar.slider("Generaciones", 10, 200, 50, step=10)
mostrar_no_factibles = st.sidebar.checkbox("Mostrar puntos no factibles", value=True)
mostrar_optimo = st.sidebar.checkbox("Marcar mejor solución", value=True)

mejor_sol = None
mejor_valor = float("inf")
def es_factible(x, pid):
    return np.all(restricciones(x, pid) <= 0)

def phi(x, pid):
    return np.sum(np.maximum(0, restricciones(x, pid)))

def evolucion_diferencial():
    global mejor_sol, mejor_valor
    poblacion = lim_inf_vec + np.random.rand(NP, D) * (lim_sup_vec - lim_inf_vec)
    historial = []

    for gen in range(generaciones):
        # Paso 1: Mutación
        mutados = np.zeros_like(poblacion)
        for i in range(NP):
            idxs = list(range(NP))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            v = poblacion[r3] + F * (poblacion[r1] - poblacion[r2])
            v = np.clip(v, lim_inf_vec, lim_sup_vec)
            mutados[i] = v

        # Paso 2: Cruce
        ensayo = np.zeros_like(poblacion)
        for i in range(NP):
            k = np.random.randint(D)
            for j in range(D):
                if np.random.rand() <= CR or j == k:
                    ensayo[i, j] = mutados[i, j]
                else:
                    ensayo[i, j] = poblacion[i, j]

        # Paso 3: Selección con reglas de Deb
        nueva_poblacion = np.zeros_like(poblacion)
        for i in range(NP):
            seleccionado = False
            factible_padre = es_factible(poblacion[i], pid)
            factible_ensayo = es_factible(ensayo[i], pid)

            if factible_padre and factible_ensayo:
                f_padre = funcion_objetivo(poblacion[i], pid)
                f_ensayo = funcion_objetivo(ensayo[i], pid)
                if f_ensayo <= f_padre:
                    nueva_poblacion[i] = ensayo[i]
                    seleccionado = True

            if not seleccionado and factible_ensayo and not factible_padre:
                nueva_poblacion[i] = ensayo[i]
                seleccionado = True

            if not seleccionado and not factible_ensayo and not factible_padre:
                phi_padre = phi(poblacion[i], pid)
                phi_ensayo = phi(ensayo[i], pid)
                if phi_ensayo <= phi_padre:
                    nueva_poblacion[i] = ensayo[i]
                    seleccionado = True

            if not seleccionado:
                nueva_poblacion[i] = poblacion[i]

        poblacion = nueva_poblacion.copy()

        # Actualizar historial de mejor solución factible
        factibles = [funcion_objetivo(ind, pid) for ind in poblacion if es_factible(ind, pid)]
        historial.append(min(factibles) if factibles else None)

        for ind in poblacion:
            if es_factible(ind, pid):
                val = funcion_objetivo(ind, pid)
                if val < mejor_valor:
                    mejor_valor = val
                    mejor_sol = ind.copy()

    return historial, poblacion

historial, poblacion_final = evolucion_diferencial()

# Gráfica de evolución
st.subheader(" Evolución de la solución")
fig, ax = plt.subplots()
valores_limpios = [v for v in historial if v is not None]
ax.plot(valores_limpios, color=PALETA["linea"], label="f(x) mínimo")
if mostrar_optimo and valores_limpios:
    min_gen = np.argmin(valores_limpios)
    ax.plot(min_gen, valores_limpios[min_gen], 'o', color=PALETA["mejor_punto"], label="Mejor f(x)")
ax.set_xlabel("Generación")
ax.set_ylabel("f(x)")
ax.set_title("Optimización con DE")
ax.grid(True)
ax.legend()
st.pyplot(fig)

if mejor_sol is not None:
    st.success(f" Mejor solución encontrada: x = {np.round(mejor_sol, 4).tolist()}, f(x) = {mejor_valor:.6f}")


# Diccionario de bloques de código
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
    trial = np.copy(poblacion)
    for i in range(NP):
        k = np.random.randint(dim)
        for j in range(dim):
            if np.random.rand() <= CR or j == k:
                trial[i, j] = mutados[i, j]
    return trial
""",
    "Selección": """
def seleccion(poblacion, trial):
    nueva = np.copy(poblacion)
    for i in range(NP):
        xi = poblacion[i]
        ui = trial[i]
        if es_factible(ui) and es_factible(xi):
            if funcion_objetivo(ui, pid) <= funcion_objetivo(xi, pid):
                nueva[i] = ui
        elif es_factible(ui):
            nueva[i] = ui
        elif not es_factible(xi):
            if phi(ui) <= phi(xi):
                nueva[i] = ui
    return nueva
"""
}

# Interfaz visual
with col1:
    st.subheader(" Pseudocódigo ")
    opcion = st.radio("Selecciona un bloque:", list(bloques.keys()), horizontal=True)

    if opcion == "Inicialización":
        st.latex(r"x_{j,i,0} = rnd_j[0,1] \cdot (x_j^U - x_j^L) + x_j^L")
        st.markdown(r"""
        Esta fórmula genera la **población inicial** de individuos distribuidos aleatoriamente dentro del espacio de búsqueda.

        ####  Variables:
        - **$x_{j,i,0}$**: Valor de la variable $j$ del individuo $i$ en la generación $0$.
        - **$rnd_j[0,1]$**: Número aleatorio uniforme en $[0,1]$ específico para la dimensión $j$.
        - **$x_j^U$, $x_j^L$**: Límites superior e inferior del dominio de la variable $x_j$.

        La fórmula asegura que **cada componente de cada individuo** se inicialice **dentro de su dominio permitido**.
        """)

    elif opcion == "Mutación":
        st.latex(r"v_{j,i,G+1} = x_{j,r3,G} + F \cdot (x_{j,r1,G} - x_{j,r2,G})")
        st.markdown(r"""
        Esta operación genera un **vector mutado** a partir de tres individuos seleccionados aleatoriamente.

        ####  Variables:
        - **$v_{j,i,G+1}$**: Componente $j$ del vector mutado del individuo $i$ en la generación $G+1$.
        - **$x_{j,r1,G}$, $x_{j,r2,G}$, $x_{j,r3,G}$**: Componentes del vector de individuos distintos a $i$.
        - **$F$**: Factor de escala que controla la magnitud del diferencial entre individuos.

        El vector mutado representa una **exploración dirigida** del espacio de búsqueda basada en diferencias entre soluciones.
        """)

    elif opcion == "Cruce":
        st.latex(r"u_{j,i,G+1} = \begin{cases} v_{j,i,G+1} & \text{si } rnd[0,1] \leq CR \lor j = k \\ x_{j,i,G} & \text{en otro caso} \end{cases}")
        st.markdown(r"""
        El **cruce binomial** mezcla el vector mutado con el vector actual para formar un **vector de prueba**.

        ####  Variables:
        - **$u_{j,i,G+1}$**: Componente del nuevo individuo de prueba.
        - **$v_{j,i,G+1}$**: Valor mutado.
        - **$x_{j,i,G}$**: Valor original previo al cruce.
        - **$CR$**: Tasa de cruce (crossover rate), define la probabilidad de heredar el gen mutado.
        - **$k$**: Índice aleatorio que garantiza que al menos una componente venga del mutado.

        Este mecanismo fomenta la **variación controlada** mientras mantiene parte de la estructura del individuo original.
        """)

    elif opcion == "Selección":
        st.latex(r"x_i^{G+1} = \begin{cases} u_i & \text{si cumple criterios de Deb} \\ x_i & \text{en otro caso} \end{cases}")
        st.markdown(r"""
        La selección decide si el individuo de prueba **reemplaza** al actual, con base en las **reglas de Deb** para problemas con restricciones.

        #### Reglas de Deb:
        1. Si **ambos** ($u_i$, $x_i$) son factibles: elige el que tenga menor $f(x)$.
        2. Si **uno solo** es factible: selecciona el factible.
        3. Si **ambos son inviables**: selecciona el que tenga menor penalización $\phi(x)$.

        """)
with col2:
    if opcion == "Inicialización":
        st.subheader(" Paso 1: Inicialización de la población")
        st.code("""
def inicializar_poblacion():
    return lim_inf + np.random.rand(NP, dim) * (lim_sup - lim_inf)
        """, language="python")

    elif opcion == "Mutación":
        st.subheader(" Paso 2: Mutación diferencial")
        st.code("""
def mutacion(poblacion):
    mutados = np.zeros_like(poblacion)
    for i in range(NP):
        idxs = list(range(NP))
        idxs.remove(i)
        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
        v = poblacion[r3] + F * (poblacion[r1] - poblacion[r2])
        mutados[i] = np.clip(v, lim_inf, lim_sup)
    return mutados
        """, language="python")

    elif opcion == "Cruce":
        st.subheader(" Paso 3: Cruce")
        st.code("""
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
        """, language="python")

    elif opcion == "Selección":
        st.subheader(" Paso 4: Selección con reglas de Deb")
        st.code("""
def seleccion(poblacion, ensayo, pid):
    NP = poblacion.shape[0]
    nueva_poblacion = np.zeros_like(poblacion)

    for i in range(NP):
        factible_padre = es_factible(poblacion[i], pid)
        factible_ensayo = es_factible(ensayo[i], pid)
        seleccionado = False

        if factible_padre and factible_ensayo:
            f_padre = funcion_objetivo(poblacion[i], pid)
            f_ensayo = funcion_objetivo(ensayo[i], pid)
            if f_ensayo <= f_padre:
                nueva_poblacion[i] = ensayo[i]
                seleccionado = True

        if not seleccionado and factible_ensayo and not factible_padre:
            nueva_poblacion[i] = ensayo[i]
            seleccionado = True

        if not seleccionado and not factible_ensayo and not factible_padre:
            phi_padre = phi(poblacion[i], pid)
            phi_ensayo = phi(ensayo[i], pid)
            if phi_ensayo <= phi_padre:
                nueva_poblacion[i] = ensayo[i]
                seleccionado = True

        if not seleccionado:
            nueva_poblacion[i] = poblacion[i]

    return nueva_poblacion
        """, language="python")
