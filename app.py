import streamlit as st
from paginas import Evolucion_Diferencial, epsilon, ranking, wolf, whale, cuckoo, resumen, ponderada, epsilon2, progMetas, obtMetas, Lexicografico

st.set_page_config(page_title="Guía DEB", layout="wide")

# Inicializar session state para controlar la navegación
if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = "Inicio"
if 'algoritmo_seleccionado' not in st.session_state:
    st.session_state.algoritmo_seleccionado = None

# Estilo para pantalla de inicio
def estilo_inicio():
    st.markdown("""
        <style>
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            text-align: center;
            padding-top: 2rem;
        }
        .titulo {
            font-size: 3rem;
            color: #f0f0f0;
            margin-bottom: 0.5rem;
        }
        .subtitulo {
            font-size: 1.4rem;
            color: #cccccc;
            margin-bottom: 2rem;
        }
        .button-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            margin-top: 0.5rem;
        }
        .stButton > button {
            width: 350px !important;
            height: 60px !important;
            font-size: 1.1rem !important;
            border-radius: 5px !important;
            border: 1px solid #ccc !important;
            background-color: #e5e5e5 !important;
            color: #333 !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
            display: block !important;
            margin: 0 auto !important;
        }
        .stButton > button:hover {
            background-color: #d1d1d1 !important;
            border-color: #999 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Función para mostrar la pantalla de inicio
def mostrar_inicio():
    estilo_inicio()
    
    # Centrar los botones verticalmente
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("""
            <div class="main-container">
                <div class="titulo">
                    <strong>Algoritmos de Optimización</strong>
                </div>
                <div class="subtitulo">
                    Selecciona el tipo de algoritmo
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        
        if st.button("Algoritmos Monoobjetivo", key="mono"):
            st.session_state.pagina_actual = "Algoritmos monoobjetivo"
            st.rerun()
        
        if st.button("Algoritmos Multiobjetivo", key="multi"):
            st.session_state.pagina_actual = "Algoritmos multiobjetivo"
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

# Función para mostrar algoritmos monoobjetivo
def mostrar_monoobjetivo():
    st.sidebar.markdown("---")
    algoritmo = st.sidebar.radio("Selecciona un algoritmo:", [
        "Evolución Diferencial",
        "Grey Wolf",
        "Whales",
        "Cuckoo",
        "Resumen"
    ])
    
    if algoritmo == "Evolución Diferencial":
        Evolucion_Diferencial.mostrar()
    elif algoritmo == "Grey Wolf":
        wolf.mostrar()
    elif algoritmo == "Whales":
        whale.mostrar()
    elif algoritmo == "Cuckoo":
        cuckoo.mostrar()
    elif algoritmo == "Resumen":
        resumen.mostrar()

# Función para mostrar algoritmos multiobjetivo
def mostrar_multiobjetivo():
    st.sidebar.markdown("---")
    algoritmo = st.sidebar.radio("Selecciona un algoritmo:", [
        "Suma Ponderada",
        "Epsilon Constraint",
        "Programacion por Metas",
        "Obtencion de Metas",
        "Lexicográfico"
    ])
    
    if algoritmo == "Suma Ponderada":
        ponderada.mostrar()
    elif algoritmo == "Epsilon Constraint":
        epsilon2.mostrar()
    elif algoritmo == "Programacion por Metas":
        progMetas.mostrar()
    elif algoritmo == "Obtencion de Metas":
        obtMetas.mostrar()
    elif algoritmo == "Lexicográfico":
        Lexicografico.mostrar()
    



# Mostrar página actual basada en session state
if st.session_state.pagina_actual == "Inicio":
    mostrar_inicio()
elif st.session_state.pagina_actual == "Algoritmos monoobjetivo":
    # Menú lateral solo para algoritmos monoobjetivo
    st.sidebar.markdown("## Navegación")
    if st.sidebar.button("Inicio"):
        st.session_state.pagina_actual = "Inicio"
        st.rerun()
    mostrar_monoobjetivo()
elif st.session_state.pagina_actual == "Algoritmos multiobjetivo":
    # Menú lateral solo para algoritmos multiobjetivo
    st.sidebar.markdown("## Navegación")
    if st.sidebar.button("Inicio"):
        st.session_state.pagina_actual = "Inicio"
        st.rerun()
    mostrar_multiobjetivo()