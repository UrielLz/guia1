import streamlit as st
from paginas import Evolucion_Diferencial, epsilon, ranking, wolf, whale, cuckoo, resumen

# Configuración general de la app
st.set_page_config(page_title="Guía DEB", layout="wide")

# Menú lateral
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Selecciona una ventana:", [
    "Evolución Diferencial", 
    "Epsilon-Constraint",
    "Ranking",
    "Grey Wolf",
    "Whales",
    "Cuckoo",
    "Resumen"
])

# Redireccionamiento a cada ventana
if pagina == "Evolución Diferencial":
    Evolucion_Diferencial.mostrar()
elif pagina == "Epsilon-Constraint":
    epsilon.mostrar()
elif pagina == "Ranking":
    ranking.mostrar()
elif pagina == "Grey Wolf":
    wolf.mostrar()
elif pagina == "Whales":
    whale.mostrar()
elif pagina == "Cuckoo":
    cuckoo.mostrar()
elif pagina == "Resumen":
    resumen.mostrar()
