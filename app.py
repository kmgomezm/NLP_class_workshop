import streamlit as st

st.set_page_config(
    page_title="Taller NLP & LLMs - EAFIT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🧠 Taller NLP & LLMs")
st.sidebar.markdown("**Maestría en Ciencia de Datos | EAFIT**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "📝 Quiz Conceptual",
        "🔤 Tokenización & Encoding",
        "📊 Vectorización Clásica",
        "🔗 Modelado de Secuencias",
        "🤖 Laboratorio LLM",
        "💬 Agente Conversacional"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Docente:** Jorge Iván Padilla-Buriticá")
st.sidebar.markdown("**Periodo:** 2026-1")

# Route to pages
if page == "🏠 Inicio":
    from pages_modules import home
    home.show()
elif page == "📝 Quiz Conceptual":
    from pages_modules import quiz
    quiz.show()
elif page == "🔤 Tokenización & Encoding":
    from pages_modules import tokenization
    tokenization.show()
elif page == "📊 Vectorización Clásica":
    from pages_modules import vectorization
    vectorization.show()
elif page == "🔗 Modelado de Secuencias":
    from pages_modules import sequences
    sequences.show()
elif page == "🤖 Laboratorio LLM":
    from pages_modules import llm_lab
    llm_lab.show()
elif page == "💬 Agente Conversacional":
    from pages_modules import agent
    agent.show()
