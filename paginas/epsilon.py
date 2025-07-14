import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from funciones import funcion_objetivo, restricciones, obtener_parametros

def mostrar():
    PALETA = {
        "linea": "#6C63FF",
        "mejor_punto": "#FF4081",
        "factibles": "#00E676",
        "no_factibles": "#FF1744",
        "optimo": "#40C4FF"
    }

    st.markdown("""
    <div style='background-color:#1E1E1E;padding:20px;border-radius:10px'>
        <h1 style='color:white;text-align:center;'>Evolución Diferencial con Epsilon-Constraint</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    st.sidebar.title("Configuración del problema")

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
    epsilon_val = st.sidebar.slider("Umbral epsilon (ε)", 0.0, 100.0, 1.0, step=0.1)
    mostrar_no_factibles = st.sidebar.checkbox("Mostrar puntos no factibles", value=True)
    mostrar_optimo = st.sidebar.checkbox("Marcar mejor solución", value=True)

    def phi(x):
        return np.sum(np.maximum(0, restricciones(x, pid)))

    def evolucion_diferencial_epsilon():
        mejor_sol = None
        mejor_valor = float("inf")
        poblacion = lim_inf_vec + np.random.rand(NP, D) * (lim_sup_vec - lim_inf_vec)
        historial = []

        for _ in range(generaciones):
            mutados = np.zeros_like(poblacion)
            for i in range(NP):
                idxs = list(range(NP))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                v = poblacion[r3] + F * (poblacion[r1] - poblacion[r2])
                mutados[i] = np.clip(v, lim_inf_vec, lim_sup_vec)

            ensayo = np.zeros_like(poblacion)
            for i in range(NP):
                k = np.random.randint(D)
                for j in range(D):
                    ensayo[i, j] = mutados[i, j] if np.random.rand() <= CR or j == k else poblacion[i, j]

            nueva_poblacion = np.zeros_like(poblacion)
            for i in range(NP):
                if phi(ensayo[i]) <= epsilon_val:
                    if phi(poblacion[i]) <= epsilon_val:
                        if funcion_objetivo(ensayo[i], pid) <= funcion_objetivo(poblacion[i], pid):
                            nueva_poblacion[i] = ensayo[i]
                        else:
                            nueva_poblacion[i] = poblacion[i]
                    else:
                        nueva_poblacion[i] = ensayo[i]
                else:
                    nueva_poblacion[i] = poblacion[i]

            poblacion = nueva_poblacion.copy()
            factibles = [funcion_objetivo(ind, pid) for ind in poblacion if phi(ind) <= epsilon_val]
            historial.append(min(factibles) if factibles else None)

            for ind in poblacion:
                if phi(ind) <= epsilon_val:
                    val = funcion_objetivo(ind, pid)
                    if val < mejor_valor:
                        mejor_valor = val
                        mejor_sol = ind.copy()

        return historial, poblacion, mejor_sol, mejor_valor

    historial, poblacion_final, mejor_sol, mejor_valor = evolucion_diferencial_epsilon()

    st.subheader(" Evolución de la solución")
    fig, ax = plt.subplots()
    valores_limpios = [v for v in historial if v is not None]
    ax.plot(valores_limpios, color=PALETA["linea"], label="f(x) mínimo")
    if mostrar_optimo and valores_limpios:
        min_gen = np.argmin(valores_limpios)
        ax.plot(min_gen, valores_limpios[min_gen], 'o', color=PALETA["mejor_punto"], label="Mejor f(x)")
    ax.set_xlabel("Generación")
    ax.set_ylabel("f(x)")
    ax.set_title("Optimización con ε-Constraint")
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
def seleccion(poblacion, ensayo, epsilon):
    nueva_poblacion = np.zeros_like(poblacion)
    for i in range(NP):
        if phi(ensayo[i]) <= epsilon:
            if phi(poblacion[i]) <= epsilon:
                if funcion_objetivo(ensayo[i], pid) <= funcion_objetivo(poblacion[i], pid):
                    nueva_poblacion[i] = ensayo[i]
                else:
                    nueva_poblacion[i] = poblacion[i]
            else:
                nueva_poblacion[i] = ensayo[i]
        else:
            nueva_poblacion[i] = poblacion[i]
    return nueva_poblacion
"""
    }

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
            st.latex(r"x_i^{G+1} = \begin{cases} u_i & \text{si } \phi(u_i) \leq \varepsilon \text{ y } f(u_i) \leq f(x_i) \\ x_i & \text{en otro caso} \end{cases}")
            st.markdown(r"""
            En esta etapa, el individuo de prueba $u_i$ puede **reemplazar** al individuo actual $x_i$ dependiendo de dos condiciones:

            ---
            ### Condiciones para aceptar $u_i$:
            1. **Cumple con el umbral epsilon**:  
               $\phi(u_i) \leq \varepsilon$  
               Es decir, el incumplimiento de las restricciones es pequeña o nula.
            
            2. **Mejora la función objetivo**:  
               $f(u_i) \leq f(x_i)$  
               Lo que implica una solución igual o mejor que la anterior.

            ---
            ### ¿Qué pasa si no se cumplen?
            - Si $u_i$ **no cumple** con el umbral o **no mejora**, entonces se **mantiene $x_i$** en la siguiente generación.

            ---
            """)


    with col2:
        st.subheader(f" Paso de {opcion}")
        st.code(bloques[opcion], language="python")
