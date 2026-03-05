import streamlit as st

QUESTIONS = [
    {
        "id": 1,
        "pregunta": "¿Cuál es la diferencia fundamental entre One-Hot Encoding y Embeddings densos?",
        "respuesta_correcta": "B",
        "opciones": {
            "A": "One-Hot captura semántica; los Embeddings no.",
            "B": "One-Hot genera vectores dispersos de alta dimensión sin semántica; los Embeddings son densos, de baja dimensión y capturan relaciones semánticas.",
            "C": "Ambos tienen la misma dimensionalidad.",
            "D": "Los Embeddings siempre requieren más memoria que One-Hot."
        },
        "explicacion": """
**One-Hot Encoding:**
- Dimensionalidad = tamaño del vocabulario (puede ser 50,000+)
- Vectores **dispersos** (casi todos ceros)
- **Sin semántica**: "rey" y "reina" son ortogonales

**Embeddings Densos (Word2Vec, GloVe, FastText):**
- Dimensionalidad baja y fija (ej. 100, 300 dimensiones)
- Vectores **densos** (todos los valores son relevantes)
- **Capturan semántica**: `rey - hombre + mujer ≈ reina`
        """
    },
    {
        "id": 2,
        "pregunta": "¿Por qué un modelo CRF suele superar a un HMM en tareas de NER?",
        "respuesta_correcta": "C",
        "opciones": {
            "A": "Porque CRF es más rápido de entrenar.",
            "B": "Porque HMM no puede manejar datos de texto.",
            "C": "CRF es un modelo discriminativo que condiciona toda la secuencia de observaciones, permitiendo features arbitrarios. HMM es generativo y asume independencia condicional entre observaciones.",
            "D": "CRF requiere menos datos de entrenamiento."
        },
        "explicacion": """
**HMM (Hidden Markov Model) — Modelo Generativo:**
- Modela P(observaciones | estados ocultos)
- Asume **independencia condicional** entre observaciones
- Solo puede usar features locales (palabra actual)

**CRF (Conditional Random Fields) — Modelo Discriminativo:**
- Modela directamente P(etiquetas | toda la secuencia)
- **Sin supuesto de independencia**: ve el contexto completo
- Acepta **features arbitrarias**: sufijos, prefijos, capitalización, vecinos
- Resultado: superior en NER donde el contexto es crucial
        """
    },
    {
        "id": 3,
        "pregunta": "¿Cómo mitigan LSTM y GRU el problema del desvanecimiento del gradiente en las RNN?",
        "respuesta_correcta": "A",
        "opciones": {
            "A": "Mediante mecanismos de compuerta (gates) que regulan el flujo de información, permitiendo que los gradientes fluyan sin degradarse a través del tiempo.",
            "B": "Usando funciones de activación ReLU en lugar de tanh.",
            "C": "Aumentando el tamaño del batch de entrenamiento.",
            "D": "Reduciendo el número de capas de la red."
        },
        "explicacion": """
**Vanishing Gradient en RNN:**
- Al retropropagar por muchos pasos temporales, el gradiente se **multiplica** repetidamente por pesos < 1
- El gradiente se vuelve exponencialmente pequeño → la red no aprende dependencias largas

**LSTM — Solución con 3 compuertas:**
- **Forget gate**: decide qué olvidar del estado de celda
- **Input gate**: decide qué nueva información guardar
- **Output gate**: decide qué parte del estado de celda exponer
- El **cell state** actúa como una "autopista" para el gradiente

**GRU — Versión simplificada:**
- Solo 2 compuertas: **Reset** y **Update**
- Menos parámetros, similar rendimiento en muchas tareas
        """
    },
    {
        "id": 4,
        "pregunta": "En LLMs, ¿cómo afecta una temperatura cercana a 2.0 la distribución Softmax de tokens?",
        "respuesta_correcta": "D",
        "opciones": {
            "A": "Hace que el modelo siempre elija el token más probable.",
            "B": "Reduce la longitud de las respuestas generadas.",
            "C": "No tiene ningún efecto sobre la distribución de probabilidades.",
            "D": "Aplana la distribución Softmax, asignando probabilidades más similares a todos los tokens, aumentando la aleatoriedad y creatividad pero reduciendo la coherencia."
        },
        "explicacion": """
**Fórmula con temperatura T:**

`P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)`

| Temperatura | Efecto |
|------------|--------|
| T → 0.0 | Distribución **punteaguda** → siempre el token más probable (greedy) |
| T = 1.0 | Distribución **original** del modelo |
| T → 2.0 | Distribución **aplanada** → todos los tokens equiprobables → máxima aleatoriedad |

**Práctica:** 
- T ≈ 0.2: Respuestas factuales, código
- T ≈ 0.7-0.9: Balance creatividad/coherencia  
- T ≈ 1.5+: Escritura creativa, poesía, brainstorming
        """
    }
]

def show():
    st.title("📝 Parte 01: Evaluación Conceptual")
    st.markdown("Responde las siguientes preguntas de selección múltiple. Luego podrás ver la explicación detallada.")
    st.markdown("---")

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    for q in QUESTIONS:
        with st.expander(f"**Pregunta {q['id']}:** {q['pregunta']}", expanded=True):
            answer = st.radio(
                "Selecciona tu respuesta:",
                list(q["opciones"].keys()),
                format_func=lambda x, q=q: f"{x}) {q['opciones'][x]}",
                key=f"q{q['id']}",
                index=None
            )
            if answer:
                st.session_state.quiz_answers[q["id"]] = answer

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submit = st.button("✅ Verificar Respuestas", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reiniciar Quiz", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()

    if submit:
        st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        st.markdown("---")
        st.markdown("## 📊 Resultados")

        score = sum(
            1 for q in QUESTIONS
            if st.session_state.quiz_answers.get(q["id"]) == q["respuesta_correcta"]
        )
        total = len(QUESTIONS)

        col1, col2 = st.columns(2)
        with col1:
            if score == total:
                st.success(f"🎉 ¡Perfecto! {score}/{total} respuestas correctas")
            elif score >= total // 2:
                st.warning(f"📚 Buen intento: {score}/{total} respuestas correctas")
            else:
                st.error(f"❌ Necesitas repasar: {score}/{total} respuestas correctas")

        st.markdown("---")
        st.markdown("## 📖 Explicaciones Detalladas")
        for q in QUESTIONS:
            user_ans = st.session_state.quiz_answers.get(q["id"])
            is_correct = user_ans == q["respuesta_correcta"]
            icon = "✅" if is_correct else "❌"
            with st.expander(f"{icon} Pregunta {q['id']}: {q['pregunta']}"):
                if user_ans:
                    if is_correct:
                        st.success(f"Tu respuesta: **{user_ans}** — ¡Correcto!")
                    else:
                        st.error(f"Tu respuesta: **{user_ans}** — Incorrecto")
                        st.info(f"Respuesta correcta: **{q['respuesta_correcta']}**")
                else:
                    st.warning("No respondiste esta pregunta.")
                    st.info(f"Respuesta correcta: **{q['respuesta_correcta']}**")
                st.markdown("**Explicación:**")
                st.markdown(q["explicacion"])
