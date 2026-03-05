import streamlit as st
import time
import plotly.graph_objects as go
import numpy as np

def get_groq_client(api_key):
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        return None

def call_groq(client, prompt, model, temperature, top_p, max_tokens=500):
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    content = response.choices[0].message.content
    usage = response.usage
    return content, latency, usage

def softmax_with_temp(logits, temperature):
    if temperature < 0.01:
        temperature = 0.01
    scaled = np.array(logits) / temperature
    scaled -= np.max(scaled)
    exp_vals = np.exp(scaled)
    return exp_vals / exp_vals.sum()

def show():
    st.title("🤖 Laboratorio de LLM")
    st.markdown("Experimenta con hiperparámetros de Llama 3.3 vía Groq API.")
    st.markdown("---")

    # API Key
    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        api_key = st.text_input("API Key:", type="password", key="llm_lab_key",
                                 placeholder="gsk_...")
        model = st.selectbox("Modelo:", [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768"
        ])

    tab1, tab2, tab3 = st.tabs(["🎛️ Playground de Temperatura", "📐 Visualización Softmax", "📊 Comparativa de Parámetros"])

    with tab1:
        st.markdown("## 🎛️ Playground: Temperatura y Top-p")

        col1, col2 = st.columns([1, 2])
        with col1:
            temperature = st.slider("🌡️ Temperatura:", 0.0, 2.0, 0.7, 0.1)
            top_p = st.slider("🎯 Top-p (nucleus sampling):", 0.01, 1.0, 0.9, 0.01)
            max_tokens = st.slider("📏 Max tokens:", 50, 500, 200, 50)

            st.markdown("---")
            st.markdown(f"""
            **Interpretación:**
            - T = {temperature:.1f}: {'Determinista' if temperature < 0.3 else 'Balanceado' if temperature < 1.0 else 'Creativo' if temperature < 1.5 else 'Muy aleatorio'}
            - top_p = {top_p:.2f}: {'Solo tokens más probables' if top_p < 0.5 else 'Balance' if top_p < 0.9 else 'Amplio rango de tokens'}
            """)

        with col2:
            prompts_template = {
                "Explicación técnica": "Explica brevemente qué es un embedding en NLP.",
                "Historia creativa": "Escribe el inicio de una historia de ciencia ficción sobre IA.",
                "Receta de cocina": "Dame una receta creativa con aguacate.",
                "Poema": "Escribe un poema corto sobre el aprendizaje automático.",
                "Personalizado": ""
            }
            prompt_type = st.selectbox("Tipo de prompt:", list(prompts_template.keys()))
            if prompt_type == "Personalizado":
                prompt = st.text_area("Tu prompt:", height=80)
            else:
                prompt = st.text_area("Prompt:", value=prompts_template[prompt_type], height=80)

        if st.button("🚀 Generar respuesta", type="primary", use_container_width=True):
            if not api_key:
                st.error("⚠️ Ingresa tu API Key de Groq en la barra lateral.")
            elif not prompt.strip():
                st.warning("Ingresa un prompt.")
            else:
                client = get_groq_client(api_key)
                if client is None:
                    st.error("No se pudo inicializar el cliente Groq. Verifica que `groq` esté instalado.")
                else:
                    with st.spinner("Generando respuesta..."):
                        try:
                            content, latency, usage = call_groq(
                                client, prompt, model, temperature, top_p, max_tokens
                            )
                            st.session_state["last_lab_response"] = {
                                "content": content,
                                "latency": latency,
                                "usage": usage,
                                "temperature": temperature,
                                "top_p": top_p,
                                "model": model
                            }
                        except Exception as e:
                            st.error(f"Error en la API: {e}")

        if "last_lab_response" in st.session_state:
            r = st.session_state["last_lab_response"]
            st.markdown("---")
            st.markdown("### 📤 Respuesta")
            st.markdown(r["content"])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Latencia", f"{r['latency']:.2f}s")
            with col2:
                total_tokens = r["usage"].total_tokens
                tps = total_tokens / r["latency"] if r["latency"] > 0 else 0
                st.metric("⚡ TPS", f"{tps:.0f}")
            with col3:
                st.metric("📝 Tokens", r["usage"].total_tokens)
            with col4:
                st.metric("🌡️ Temp usada", r["temperature"])

    with tab2:
        st.markdown("## 📐 Visualización del efecto de la Temperatura en Softmax")
        st.markdown("""
        La temperatura **T** escala los logits antes de aplicar Softmax:
        
        `P(token_i) = exp(logit_i / T) / Σ_j exp(logit_j / T)`
        """)

        temp_viz = st.slider("🌡️ Temperatura para visualizar:", 0.1, 2.0, 1.0, 0.1, key="temp_viz")

        # Simulate 10 tokens with different logits
        token_names = ["el", "la", "un", "de", "en", "que", "y", "los", "se", "por"]
        raw_logits = [3.2, 2.8, 2.1, 1.9, 1.5, 1.2, 0.9, 0.7, 0.4, 0.1]

        probs_t1 = softmax_with_temp(raw_logits, 1.0)
        probs_t = softmax_with_temp(raw_logits, temp_viz)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="T = 1.0 (base)",
            x=token_names,
            y=probs_t1,
            marker_color="#3b82f6",
            opacity=0.6
        ))
        fig.add_trace(go.Bar(
            name=f"T = {temp_viz}",
            x=token_names,
            y=probs_t,
            marker_color="#ef4444"
        ))
        fig.update_layout(
            barmode="group",
            title=f"Distribución de probabilidad con T=1.0 vs T={temp_viz}",
            xaxis_title="Token",
            yaxis_title="Probabilidad",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        entropy_base = -np.sum(probs_t1 * np.log(probs_t1 + 1e-10))
        entropy_t = -np.sum(probs_t * np.log(probs_t + 1e-10))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entropía base (T=1)", f"{entropy_base:.3f} bits")
        with col2:
            st.metric(f"Entropía (T={temp_viz})", f"{entropy_t:.3f} bits")
        with col3:
            max_prob_t = max(probs_t)
            st.metric("P(token más probable)", f"{max_prob_t:.3f}")

        if temp_viz < 0.5:
            st.warning(f"⚠️ T={temp_viz}: Distribución muy concentrada. El modelo es casi determinista.")
        elif temp_viz > 1.5:
            st.info(f"🎨 T={temp_viz}: Distribución muy plana. Alta aleatoriedad y creatividad.")
        else:
            st.success(f"✅ T={temp_viz}: Buen balance entre diversidad y coherencia.")

    with tab3:
        st.markdown("## 📊 Guía de Hiperparámetros")

        data = {
            "Parámetro": ["Temperatura", "Top-p", "Top-k", "Max Tokens", "Frequency Penalty", "Presence Penalty"],
            "Rango típico": ["0.0 – 2.0", "0.01 – 1.0", "1 – 100", "1 – 4096", "-2.0 – 2.0", "-2.0 – 2.0"],
            "Efecto bajo": [
                "Determinista, repetitivo",
                "Solo tokens más probables",
                "Solo 1 token",
                "Respuestas cortas",
                "Sin penalización",
                "Sin penalización"
            ],
            "Efecto alto": [
                "Muy aleatorio, incoherente",
                "Amplia distribución",
                "100 tokens posibles",
                "Respuestas largas",
                "Evita repetición de palabras",
                "Promueve temas nuevos"
            ],
            "Recomendado (factual)": ["0.1-0.3", "0.5-0.7", "10-20", "256-512", "0.0-0.3", "0.0"],
            "Recomendado (creativo)": ["0.8-1.2", "0.9-1.0", "50-100", "500-2000", "0.3-0.5", "0.3-0.5"]
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
