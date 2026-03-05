import streamlit as st

def show():
    st.title("🧠 Taller de Entrenamiento: NLP y LLMs")
    st.markdown("### Experimentación con Groq, Llama 3.3 y Arquitecturas Secuenciales")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Curso:** Inteligencia Artificial ECA&I (Posgrado)")
    with col2:
        st.info("**Docente:** Jorge Iván Padilla-Buriticá")
    with col3:
        st.info("**Universidad:** EAFIT - Periodo 2026-1")

    st.markdown("---")
    st.markdown("## 🎯 Objetivo del Taller")
    st.markdown("""
    Desarrollar una aplicación interactiva en Streamlit que permita visualizar la **evolución del 
    Procesamiento de Lenguaje Natural (NLP)**. El estudiante transitará desde:

    - 📐 **Representaciones vectoriales clásicas** (BoW, TF-IDF)
    - 🔗 **Modelos probabilísticos de secuencia** (HMM, CRF)
    - 🤖 **Agente Conversacional** basado en Llama 3.3 (Groq API)
    
    Evaluando el impacto de hiperparámetros como la **temperatura** y el **top-p**.
    """)

    st.markdown("---")
    st.markdown("## 📋 Estructura del Taller")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Parte 01 — Quiz Conceptual
        - One-Hot vs Embeddings densos
        - CRF vs HMM en NER
        - Vanishing gradient en RNNs
        - Temperatura en LLMs

        ### Parte 02 — Implementación
        - Tokenización & Encoding
        - Vectorización Clásica (BoW, TF-IDF)
        - Modelado de Secuencias (N-grams, RNN/LSTM/GRU)
        """)
    with col2:
        st.markdown("""
        ### Laboratorio LLM
        - Playground de temperatura (0.0 → 2.0)
        - Observar creatividad vs coherencia

        ### Parte 03 — Agente Conversacional
        - Agente especializado con Llama 3.3
        - Métricas: Latencia, Auto-evaluación, TPS
        """)

    st.markdown("---")
    st.markdown("## ⚙️ Configuración de API Key")
    st.markdown("""
    Para usar las funcionalidades de LLM (Laboratorio y Agente), necesitas una **Groq API Key**:
    
    1. Regístrate en [console.groq.com](https://console.groq.com)
    2. Genera tu API Key gratuita
    3. Ingrésala en las secciones **Laboratorio LLM** o **Agente Conversacional**
    
    > 🔒 La clave nunca se almacena — solo se usa durante la sesión activa.
    """)

    st.markdown("---")
    st.markdown("> *\"El lenguaje es el vestido del pensamiento.\"* — Samuel Johnson")
