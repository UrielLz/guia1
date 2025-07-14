import streamlit as st

def mostrar():
    st.set_page_config(layout="wide")
    
    st.title("Resumen")
    
    # Diccionario de algoritmos simplificado
    algoritmos = {
        "Evolución Diferencial con manejo de restricciones (Deb)": {
            "descripcion": "Algoritmo basado en poblaciones que utiliza operadores de mutación, cruce y selección para explorar el espacio de soluciones. Las restricciones se manejan con las reglas de factibilidad de Deb.",
            "pseudocodigo": """
ALGORITMO: Evolución Diferencial con Restricciones (Deb)

ENTRADA: 
- Población inicial N
- Factor de mutación F
- Probabilidad de cruce CR
- Límites [xL, xU]

SALIDA: Mejor solución encontrada

PASO 1: INICIALIZACIÓN
    Para cada individuo i de 1 a N:
        Para cada dimensión j de 1 a D:
            x[j,i] = xL[j] + random() * (xU[j] - xL[j])
        Fin Para
    Fin Para

PASO 2: EVALUACIÓN INICIAL
    Para cada individuo i de 1 a N:
        f[i] = evaluar_función_objetivo(x[i])
        incumplimiento[i] = calcular_incumplimiento_restricciones(x[i])
    Fin Para

PASO 3: BUCLE PRINCIPAL
    Para generación = 1 hasta MAX_GENERACIONES:
        Para cada individuo i de 1 a N:
            
            // MUTACIÓN
            Seleccionar r1, r2, r3 diferentes de i
            Para cada dimensión j:
                v[j,i] = x[j,r3] + F * (x[j,r1] - x[j,r2])
            Fin Para
            
            // CRUCE
            k = entero_aleatorio(1, D)
            Para cada dimensión j:
                Si random() < CR o j == k:
                    u[j,i] = v[j,i]
                Sino:
                    u[j,i] = x[j,i]
                Fin Si
            Fin Para
            
            // REPARACIÓN DE LÍMITES
            Para cada dimensión j:
                u[j,i] = max(xL[j], min(xU[j], u[j,i]))
            Fin Para
            
            // EVALUACIÓN
            f_nuevo = evaluar_función_objetivo(u[i])
            incumplimiento_nuevo = calcular_incumplimiento_restricciones(u[i])
            
            // SELECCIÓN CON REGLAS DE DEB
            Si es_mejor_segun_deb(u[i], x[i]):
                x[i] = u[i]
                f[i] = f_nuevo
                incumplimiento[i] = incumplimiento_nuevo
            Fin Si
        Fin Para
    Fin Para

REGLAS DE DEB:
    Si incumplimiento_nuevo < incumplimiento_actual:
        return verdadero  // Nuevo es más factible
    Si incumplimiento_nuevo > incumplimiento_actual:
        return falso     // Actual es más factible
    Sino:
        return f_nuevo < f_actual  // Comparar por función objetivo
    Fin Si
            """
        },
        "Método epsilon-constraint": {
            "descripcion": "Método clásico de optimización multiobjetivo que convierte una de las funciones objetivo en función principal y las otras en restricciones controladas por epsilon.",
            "pseudocodigo": """
ALGORITMO: Método Epsilon-Constraint

ENTRADA:
- Funciones objetivo f1, f2, ..., fk
- Valor inicial epsilon
- Paso delta_epsilon

SALIDA: Conjunto de soluciones Pareto

PASO 1: INICIALIZACIÓN
    epsilon_actual = epsilon_inicial
    soluciones_pareto = []

PASO 2: BUCLE PRINCIPAL
    Mientras epsilon_actual >= epsilon_mínimo:
        
        // DEFINIR PROBLEMA TRANSFORMADO
        función_principal = f1(x)
        restricciones_epsilon = [f2(x) <= epsilon_actual, f3(x) <= epsilon_actual, ...]
        
        // RESOLVER PROBLEMA MONO-OBJETIVO
        mejor_solución = resolver_problema_mono_objetivo(función_principal, restricciones_epsilon)
        
        // GUARDAR SOLUCIÓN
        soluciones_pareto.agregar(mejor_solución)
        
        // ACTUALIZAR EPSILON
        epsilon_actual = epsilon_actual - delta_epsilon
    Fin Mientras

FUNCIÓN resolver_problema_mono_objetivo:
    Para iteración = 1 hasta MAX_ITERACIONES:
        Para cada individuo i:
            individuo_nuevo = aplicar_operador_búsqueda(individuo[i])
            
            f1_nuevo = f1(individuo_nuevo)
            incumplimiento_epsilon = calcular_incumplimiento_epsilon(individuo_nuevo, epsilon_actual)
            
            Si es_mejor_con_epsilon(individuo_nuevo, individuo[i]):
                individuo[i] = individuo_nuevo
            Fin Si
        Fin Para
    Fin Para
    return mejor_individuo

FUNCIÓN calcular_incumplimiento_epsilon:
    incumplimiento = 0
    Para i = 2 hasta k:  // f2, f3, ..., fk
        incumplimiento += max(0, fi(x) - epsilon)
    Fin Para
    return incumplimiento
            """
        },
        "Stochastic Ranking (SR)": {
            "descripcion": "Técnica que ordena soluciones estocásticamente usando una mezcla entre valor de la función objetivo y grado de incumplimiento de restricciones.",
            "pseudocodigo": """
ALGORITMO: Stochastic Ranking

ENTRADA:
- Población de soluciones
- Probabilidad Pf (típicamente 0.45)
- Número de intercambios

SALIDA: Población ordenada

PASO 1: EVALUACIÓN
    Para cada individuo i:
        f[i] = evaluar_función_objetivo(x[i])
        incumplimiento[i] = calcular_suma_incumplimientos(x[i])
        es_factible[i] = (incumplimiento[i] == 0)
    Fin Para

PASO 2: ORDENAMIENTO ESTOCÁSTICO
    Para intercambio = 1 hasta N:
        Para i = 1 hasta N-1:
            
            // DETERMINAR TIPO DE COMPARACIÓN
            Si es_factible[i] Y es_factible[i+1]:
                // Ambos factibles: comparar por función objetivo
                Si f[i] > f[i+1]:
                    intercambiar(i, i+1)
                Fin Si
                
            Sino Si NO es_factible[i] Y NO es_factible[i+1]:
                // Ambos inviables: comparación estocástica
                Si random() < Pf:
                    // Comparar por función objetivo
                    Si f[i] > f[i+1]:
                        intercambiar(i, i+1)
                    Fin Si
                Sino:
                    // Comparar por incumplimiento de restricciones
                    Si incumplimiento[i] > incumplimiento[i+1]:
                        intercambiar(i, i+1)
                    Fin Si
                Fin Si
                
            Sino:
                // Uno factible, otro inviable: factible es mejor
                Si NO es_factible[i] Y es_factible[i+1]:
                    intercambiar(i, i+1)
                Fin Si
            Fin Si
        Fin Para
    Fin Para

FUNCIÓN intercambiar(i, j):
    temp = x[i]
    x[i] = x[j]
    x[j] = temp
    // Intercambiar también fitness e incumplimientos
            """
        },
        "Grey Wolf Optimizer (GWO)": {
            "descripcion": "Algoritmo basado en la jerarquía de lobos: alfa (mejor), beta (segundo mejor) y delta (tercer mejor). Cada individuo actualiza su posición guiado por los tres líderes.",
            "pseudocodigo": """
ALGORITMO: Grey Wolf Optimizer

ENTRADA:
- Población N
- Límites [xL, xU]
- Máximo de iteraciones

SALIDA: Mejor solución (lobo alfa)

PASO 1: INICIALIZACIÓN
    Para cada lobo i de 1 a N:
        Para cada dimensión j:
            x[j,i] = xL[j] + random() * (xU[j] - xL[j])
        Fin Para
        fitness[i] = evaluar_función_objetivo(x[i])
    Fin Para

PASO 2: SELECCIÓN DE LÍDERES
    Ordenar población por fitness
    alfa = mejor_solución
    beta = segunda_mejor_solución
    delta = tercera_mejor_solución

PASO 3: BUCLE PRINCIPAL
    Para iteración = 1 hasta MAX_ITERACIONES:
        // ACTUALIZAR PARÁMETROS
        a = 2 - 2 * (iteración / MAX_ITERACIONES)  // Decrece de 2 a 0
        
        Para cada lobo i:
            Para cada dimensión j:
                // GENERAR COEFICIENTES ALEATORIOS
                r1 = random(), r2 = random(), r3 = random()
                r4 = random(), r5 = random(), r6 = random()
                
                // CALCULAR COEFICIENTES
                A1 = 2 * a * r1 - a
                A2 = 2 * a * r2 - a
                A3 = 2 * a * r3 - a
                C1 = 2 * r4
                C2 = 2 * r5
                C3 = 2 * r6
                
                // CALCULAR DISTANCIAS A LÍDERES
                D_alfa = |C1 * alfa[j] - x[j,i]|
                D_beta = |C2 * beta[j] - x[j,i]|
                D_delta = |C3 * delta[j] - x[j,i]|
                
                // CALCULAR NUEVAS POSICIONES
                X1 = alfa[j] - A1 * D_alfa
                X2 = beta[j] - A2 * D_beta
                X3 = delta[j] - A3 * D_delta
                
                // ACTUALIZAR POSICIÓN (PROMEDIO)
                x[j,i] = (X1 + X2 + X3) / 3
                
                // REPARAR LÍMITES
                x[j,i] = max(xL[j], min(xU[j], x[j,i]))
            Fin Para
            
            // EVALUAR NUEVA POSICIÓN
            fitness[i] = evaluar_función_objetivo(x[i])
        Fin Para
        
        // ACTUALIZAR LÍDERES
        Ordenar población por fitness
        alfa = mejor_solución
        beta = segunda_mejor_solución
        delta = tercera_mejor_solución
    Fin Para
            """
        },
        "Whale Optimization Algorithm (WOA)": {
            "descripcion": "Imitación del comportamiento de caza de las ballenas jorobadas usando movimientos en espiral y técnicas de rodeo.",
            "pseudocodigo": """
ALGORITMO: Whale Optimization Algorithm

ENTRADA:
- Población N
- Límites [xL, xU]
- Máximo de iteraciones

SALIDA: Mejor solución

PASO 1: INICIALIZACIÓN
    Para cada ballena i de 1 a N:
        Para cada dimensión j:
            x[j,i] = xL[j] + random() * (xU[j] - xL[j])
        Fin Para
        fitness[i] = evaluar_función_objetivo(x[i])
    Fin Para
    
    mejor_ballena = encontrar_mejor_solución()

PASO 2: BUCLE PRINCIPAL
    Para iteración = 1 hasta MAX_ITERACIONES:
        Para cada ballena i:
            // ACTUALIZAR PARÁMETROS
            a = 2 - 2 * (iteración / MAX_ITERACIONES)  // Decrece linealmente
            r1 = random(), r2 = random()
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1  // Constante para espiral
            l = 2 * random() - 1  // Número en [-1, 1]
            p = random()  // Probabilidad para elegir mecanismo
            
            Si p < 0.5:
                Si |A| < 1:
                    // FASE DE EXPLOTACIÓN: Rodear la presa
                    Para cada dimensión j:
                        D = |C * mejor_ballena[j] - x[j,i]|
                        x[j,i] = mejor_ballena[j] - A * D
                    Fin Para
                Sino:
                    // FASE DE EXPLORACIÓN: Buscar nueva presa
                    ballena_aleatoria = seleccionar_ballena_aleatoria()
                    Para cada dimensión j:
                        D = |C * ballena_aleatoria[j] - x[j,i]|
                        x[j,i] = ballena_aleatoria[j] - A * D
                    Fin Para
                Fin Si
            Sino:
                // ATAQUE EN ESPIRAL
                Para cada dimensión j:
                    D = |mejor_ballena[j] - x[j,i]|
                    x[j,i] = D * exp(b * l) * cos(2 * π * l) + mejor_ballena[j]
                Fin Para
            Fin Si
            
            // REPARAR LÍMITES
            Para cada dimensión j:
                x[j,i] = max(xL[j], min(xU[j], x[j,i]))
            Fin Para
            
            // EVALUAR NUEVA POSICIÓN
            fitness[i] = evaluar_función_objetivo(x[i])
        Fin Para
        
        // ACTUALIZAR MEJOR SOLUCIÓN
        mejor_ballena = encontrar_mejor_solución()
    Fin Para
            """
        },
        "Cuckoo Search (CS)": {
            "descripcion": "Algoritmo inspirado en el comportamiento de reproducción de los cucos y vuelos de Lévy para exploración eficiente del espacio de búsqueda.",
            "pseudocodigo": """
ALGORITMO: Cuckoo Search

ENTRADA:
- Número de nidos N
- Probabilidad de abandono pa (típicamente 0.25)
- Máximo de iteraciones

SALIDA: Mejor nido encontrado

PASO 1: INICIALIZACIÓN
    Para cada nido i de 1 a N:
        Para cada dimensión j:
            x[j,i] = xL[j] + random() * (xU[j] - xL[j])
        Fin Para
        fitness[i] = evaluar_función_objetivo(x[i])
    Fin Para
    
    mejor_nido = encontrar_mejor_solución()

PASO 2: BUCLE PRINCIPAL
    Para iteración = 1 hasta MAX_ITERACIONES:
        Para cada nido i:
            // GENERAR NUEVO HUEVO CON LÉVY FLIGHT
            x_nuevo = x[i] + alfa * levy_flight(D)
            
            // REPARAR LÍMITES
            Para cada dimensión j:
                x_nuevo[j] = max(xL[j], min(xU[j], x_nuevo[j]))
            Fin Para
            
            // EVALUAR NUEVO HUEVO
            fitness_nuevo = evaluar_función_objetivo(x_nuevo)
            
            // SELECCIÓN: Elegir nido aleatorio para comparar
            j = entero_aleatorio(1, N)
            
            // REEMPLAZAR SI ES MEJOR
            Si fitness_nuevo < fitness[j]:
                x[j] = x_nuevo
                fitness[j] = fitness_nuevo
            Fin Si
        Fin Para
        
        // ABANDONO DE NIDOS
        Para cada nido i:
            Si random() < pa:
                // GENERAR NUEVO NIDO ALEATORIO
                Para cada dimensión j:
                    x[j,i] = xL[j] + random() * (xU[j] - xL[j])
                Fin Para
                fitness[i] = evaluar_función_objetivo(x[i])
            Fin Si
        Fin Para
        
        // ACTUALIZAR MEJOR NIDO
        mejor_nido = encontrar_mejor_solución()
    Fin Para

FUNCIÓN levy_flight(D):
    beta = 1.5
    sigma = calculado según fórmula estándar de Lévy
    
    Para cada dimensión j:
        u = normal(0, sigma)
        v = normal(0, 1)
        step = u / |v|^(1/beta)
        levy[j] = step
    Fin Para
    
    return levy
            """
        }
    }
    
    # Sidebar para navegación
    with st.sidebar:
        st.header("Navegación")
        algoritmo_seleccionado = st.selectbox(
            "Selecciona un algoritmo:",
            list(algoritmos.keys()),
            index=0
        )

    # Contenido principal
    datos = algoritmos[algoritmo_seleccionado]
    
    # Título del algoritmo
    st.subheader(algoritmo_seleccionado)
    
    # Descripción
    st.markdown("**Descripción:**")
    st.write(datos['descripcion'])
    
    # Pseudocódigo
    st.markdown("**Pseudocódigo:**")
    st.code(datos['pseudocodigo'], language='text')

if __name__ == "__main__":
    mostrar()