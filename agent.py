import streamlit as st
import time

AGENT_PROFILES = {
    "🎓 Consultor Académico NLP": {
        "system": """Eres un consultor académico experto en NLP y Machine Learning. 
Tu nombre es ARIA (Asistente de Recursos en Inteligencia Artificial).
Ayudas a estudiantes de Maestría en Ciencia de Datos de EAFIT.
Responde de forma clara, didáctica y con ejemplos cuando sea posible.
Usa terminología técnica precisa pero asegúrate de que sea comprensible.
Cuando cites datos estadísticos o comparaciones de rendimiento, sé específico.""",
        "description": "Especializado en NLP, ML y recursos académicos para maestría.",
        "welcome": "¡Hola! Soy ARIA, tu consultor académico en NLP. ¿Tienes dudas sobre el curso, arquitecturas de modelos o conceptos de procesamiento de lenguaje?"
    },
    "💻 Asistente de Código Python": {
        "system": """Eres un experto en Python, especialmente en librerías de NLP y ML como:
Hugging Face, scikit-learn, NLTK, spaCy, PyTorch, TensorFlow, LangChain, Groq.
Siempre proporciona código funcional, bien comentado y siguiendo PEP8.
Cuando des código, explica qué hace cada parte.
Si detectas errores en el código del usuario, señálalos y ofrece correcciones.
Incluye ejemplos de uso y manejo de excepciones cuando sea relevante.""",
        "description": "Experto en Python, NLP libraries, ML frameworks.",
        "welcome": "¡Hola! Soy tu asistente de código Python. Puedo ayudarte con implementaciones de NLP, debugging y mejores prácticas. ¿En qué trabajamos hoy?"
    },
    "⚽ Experto en Deportes": {
        "system": """Eres un experto en análisis deportivo con un amplio conocimiento del fútbol colombiano e internacional.
Conoces estadísticas detalladas, historia, jugadores y tendencias tácticas.
Das análisis objetivos basados en datos cuando están disponibles.
Responde con entusiasmo pero mantén la precisión factual.""",
        "description": "Análisis deportivo, fútbol colombiano e internacional.",
        "welcome": "¡Bienvenido! Soy tu experto en análisis deportivo. ¿Hablamos de la Liga BetPlay, Copa Libertadores o algún otro torneo?"
    },
    "🧘 Coach de Productividad": {
        "system": """Eres un coach de productividad y bienestar para estudiantes de posgrado.
Conoces técnicas de gestión del tiempo (Pomodoro, GTD, Time Blocking), manejo del estrés académico y optimización del aprendizaje.
Das consejos prácticos, basados en evidencia científica cuando aplica.
Eres empático y motivador. Adapta tus recomendaciones al contexto de un estudiante de maestría.""",
        "description": "Gestión del tiempo, bienestar y productividad académica.",
        "welcome": "¡Hola! Soy tu coach de productividad. Estudiar una maestría puede ser intenso. ¿En qué área quieres mejorar hoy? (gestión del tiempo, concentración, reducción del estrés...)"
    }
}

def get_groq_client(api_key):
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        return None

def call_agent(client, messages, system_prompt, model, temperature, top_p):
    start = time.time()
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=model,
        messages=all_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=800,
    )
    latency = time.time() - start
    content = response.choices[0].message.content
    usage = response.usage
    return content, latency, usage

def auto_evaluate(client, user_prompt, agent_response, model):
    eval_prompt = f"""Evalúa la siguiente respuesta de un agente de IA.

Pregunta del usuario: "{user_prompt}"

Respuesta del agente: "{agent_response}"

Califica la respuesta en una escala del 1 al 10 considerando:
- Veracidad y precisión factual (4 puntos)
- Relevancia y pertinencia (3 puntos)
- Claridad y coherencia (2 puntos)
- Utilidad práctica (1 punto)

Responde ÚNICAMENTE con este formato JSON (sin texto adicional):
{{"score": <número del 1 al 10>, "justificacion": "<máximo 2 oraciones>"}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=150
        )
        import json
        text = resp.choices[0].message.content.strip()
        # Try to extract JSON
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("score", "N/A"), data.get("justificacion", "Sin justificación")
    except Exception:
        pass
    return "N/A", "Error en auto-evaluación"

def show():
    st.title("💬 Agente Conversacional")
    st.markdown("Agente especializado con Llama 3.3 vía Groq, con métricas de rendimiento.")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        api_key = st.text_input("API Key:", type="password", key="agent_key",
                                 placeholder="gsk_...")
        st.markdown("### ⚙️ Configuración")
        model = st.selectbox("Modelo:", [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
        ], key="agent_model")
        temperature = st.slider("🌡️ Temperatura:", 0.0, 1.5, 0.7, 0.1, key="agent_temp")
        top_p = st.slider("🎯 Top-p:", 0.1, 1.0, 0.9, 0.05, key="agent_top_p")
        auto_eval = st.checkbox("🔍 Auto-evaluación (LLM-as-Judge)", value=True)

    # Agent selection
    st.markdown("### 🤖 Selecciona tu Agente")
    agent_name = st.selectbox(
        "Tipo de agente:",
        list(AGENT_PROFILES.keys())
    )
    profile = AGENT_PROFILES[agent_name]
    st.info(f"**{agent_name}:** {profile['description']}")

    # Init session state
    if "agent_name" not in st.session_state or st.session_state.agent_name != agent_name:
        st.session_state.agent_name = agent_name
        st.session_state.messages = []
        st.session_state.metrics = []

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.messages = []
            st.session_state.metrics = []
            st.rerun()

    st.markdown("---")
    st.markdown("### 💬 Conversación")

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Welcome message
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.markdown(profile["welcome"])

        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "metrics" in msg:
                    m = msg["metrics"]
                    cols = st.columns(4)
                    cols[0].caption(f"⏱️ {m['latency']:.2f}s")
                    cols[1].caption(f"⚡ {m['tps']:.0f} TPS")
                    cols[2].caption(f"🪙 {m['tokens']} tokens")
                    if m.get("eval_score") and m["eval_score"] != "N/A":
                        score = m["eval_score"]
                        color = "🟢" if score >= 8 else "🟡" if score >= 5 else "🔴"
                        cols[3].caption(f"{color} Score: {score}/10")

    # Metrics Summary
    if st.session_state.metrics:
        st.markdown("---")
        st.markdown("### 📊 Métricas de la Sesión")
        metrics = st.session_state.metrics

        col1, col2, col3, col4 = st.columns(4)
        avg_latency = sum(m["latency"] for m in metrics) / len(metrics)
        avg_tps = sum(m["tps"] for m in metrics) / len(metrics)
        total_tokens = sum(m["tokens"] for m in metrics)
        valid_scores = [m["eval_score"] for m in metrics if isinstance(m.get("eval_score"), (int, float))]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        col1.metric("⏱️ Latencia promedio", f"{avg_latency:.2f}s")
        col2.metric("⚡ TPS promedio", f"{avg_tps:.0f}")
        col3.metric("🪙 Tokens totales", total_tokens)
        if avg_score:
            col4.metric("🎯 Score promedio", f"{avg_score:.1f}/10")

        # Metrics history chart
        if len(metrics) > 1:
            import plotly.graph_objects as go
            turns = list(range(1, len(metrics)+1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=turns, y=[m["latency"] for m in metrics],
                                      name="Latencia (s)", line=dict(color="#3b82f6")))
            if valid_scores:
                fig.add_trace(go.Scatter(x=turns[:len(valid_scores)], y=valid_scores,
                                          name="Score (/10)", yaxis="y2",
                                          line=dict(color="#10b981", dash="dot")))
            fig.update_layout(
                title="Métricas por turno",
                xaxis_title="Turno",
                yaxis=dict(title="Latencia (s)"),
                yaxis2=dict(title="Score", overlaying="y", side="right"),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    # Input
    user_input = st.chat_input("Escribe tu mensaje...")

    if user_input:
        if not api_key:
            st.error("⚠️ Ingresa tu API Key de Groq en la barra lateral.")
        else:
            client = get_groq_client(api_key)
            if client is None:
                st.error("No se pudo inicializar el cliente Groq.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.spinner("El agente está pensando..."):
                    try:
                        # Prepare messages for API (only role/content)
                        api_messages = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                        content, latency, usage = call_agent(
                            client, api_messages, profile["system"],
                            model, temperature, top_p
                        )

                        tps = usage.total_tokens / latency if latency > 0 else 0

                        # Auto-eval
                        eval_score, eval_just = None, None
                        if auto_eval:
                            eval_score, eval_just = auto_evaluate(client, user_input, content, model)

                        metric = {
                            "latency": latency,
                            "tps": tps,
                            "tokens": usage.total_tokens,
                            "eval_score": eval_score,
                            "eval_justification": eval_just
                        }
                        st.session_state.metrics.append(metric)

                        # Add assistant message with metrics
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": content,
                            "metrics": metric
                        })

                        if eval_score and eval_just and eval_score != "N/A":
                            st.session_state.last_eval = f"**LLM-as-Judge:** Score {eval_score}/10 — {eval_just}"

                    except Exception as e:
                        st.error(f"Error: {e}")

                st.rerun()

    if "last_eval" in st.session_state and st.session_state.last_eval:
        st.caption(st.session_state.last_eval)
