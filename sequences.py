import streamlit as st
import re
from collections import Counter, defaultdict
import random
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def tokenize(text):
    return re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())

def build_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

def show():
    st.title("🔗 Modelado de Secuencias")
    st.markdown("N-grams, análisis de dependencias y comparativa de arquitecturas secuenciales.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 N-grams", "🏗️ Arquitecturas Secuenciales", "🧮 HMM vs CRF"])

    with tab1:
        st.markdown("## 📊 Análisis de N-grams")
        st.markdown("""
        Los **N-grams** son secuencias contiguas de N elementos (palabras o caracteres).
        Son la base de los modelos de lenguaje probabilísticos clásicos.
        """)

        default_text = """El procesamiento del lenguaje natural es un campo de la inteligencia artificial.
Los modelos de lenguaje aprenden patrones estadísticos del texto.
La inteligencia artificial ha transformado la manera en que procesamos el lenguaje.
Los algoritmos de aprendizaje profundo mejoran el procesamiento del texto.
El lenguaje natural presenta desafíos únicos para los sistemas de inteligencia artificial."""

        text = st.text_area("📝 Texto para análisis de N-grams:", value=default_text, height=130)
        tokens = tokenize(text)

        if len(tokens) < 3:
            st.warning("Ingresa más texto para generar n-grams.")
            return

        st.info(f"**Tokens identificados:** {len(tokens)} | **Vocab único:** {len(set(tokens))}")

        col1, col2 = st.columns(2)
        with col1:
            n_val = st.slider("Tamaño de N-gram:", min_value=2, max_value=5, value=2)
        with col2:
            top_k = st.slider("Mostrar Top K:", min_value=5, max_value=30, value=15)

        ngrams = build_ngrams(tokens, n_val)
        freq = Counter(ngrams)
        top_ngrams = freq.most_common(top_k)

        if not top_ngrams:
            st.warning("No hay suficientes tokens para generar n-grams.")
            return

        labels = [" ".join(ng) for ng, _ in top_ngrams]
        counts = [c for _, c in top_ngrams]

        name_map = {2: "Bigramas", 3: "Trigramas", 4: "4-gramas", 5: "5-gramas"}
        fig = px.bar(
            x=counts[::-1],
            y=labels[::-1],
            orientation="h",
            title=f"Top {top_k} {name_map.get(n_val, f'{n_val}-gramas')} más frecuentes",
            labels={"x": "Frecuencia", "y": "N-gram"},
            color=counts[::-1],
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # N-gram comparison
        st.markdown("### 📈 Cobertura por tamaño de N")
        coverage_data = []
        for n in range(1, 6):
            ngs = build_ngrams(tokens, n)
            unique = len(set(ngs))
            total = len(ngs)
            coverage_data.append({"N": n, "Total": total, "Únicos": unique, "Ratio": round(unique/total, 3) if total > 0 else 0})

        df_cov = pd.DataFrame(coverage_data)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Total N-grams", x=df_cov["N"], y=df_cov["Total"], marker_color="#3b82f6"))
        fig2.add_trace(go.Bar(name="N-grams Únicos", x=df_cov["N"], y=df_cov["Únicos"], marker_color="#10b981"))
        fig2.update_layout(barmode="group", title="Total vs Únicos por tamaño de N", xaxis_title="N", yaxis_title="Cantidad")
        st.plotly_chart(fig2, use_container_width=True)

        st.info("""
        💡 **Observa:** A mayor N, hay más n-grams únicos respecto al total (ratio aumenta).
        Esto se conoce como **data sparsity** — el principal problema de los modelos de lenguaje clásicos de N-gramas.
        """)

        # Simple text generation
        st.markdown("### 🎲 Generación de Texto con N-gram Language Model")
        if st.button("Generar oración", type="secondary"):
            # Build bigram model
            bigram_model = defaultdict(list)
            bigrams = build_ngrams(tokens, 2)
            for bg in bigrams:
                bigram_model[bg[0]].append(bg[1])

            # Generate
            start_word = random.choice(tokens[:10])
            generated = [start_word]
            for _ in range(15):
                next_words = bigram_model.get(generated[-1], [])
                if not next_words:
                    break
                generated.append(random.choice(next_words))

            st.success(f"**Texto generado:** {' '.join(generated)}")

    with tab2:
        st.markdown("## 🏗️ Comparativa de Arquitecturas Secuenciales")

        # Architecture comparison table
        data = {
            "Característica": [
                "Tipo de memoria",
                "Manejo de dependencias largas",
                "Velocidad de entrenamiento",
                "Parámetros",
                "Vanishing gradient",
                "Paralelización",
                "Uso típico actual",
            ],
            "RNN": [
                "Estado oculto simple h_t",
                "❌ Pobre (>10 pasos)",
                "✅ Rápido",
                "Pocos",
                "❌ Severo",
                "❌ No (secuencial)",
                "Tareas simples, demos",
            ],
            "LSTM": [
                "Cell state + hidden state",
                "✅ Bueno (50-100 pasos)",
                "🔶 Moderado",
                "4× más que RNN",
                "✅ Muy mitigado",
                "❌ No (secuencial)",
                "NER, Sentiment, Traducción",
            ],
            "GRU": [
                "Hidden state (gates integrados)",
                "✅ Bueno (similar LSTM)",
                "✅ Más rápido que LSTM",
                "3× más que RNN",
                "✅ Mitigado",
                "❌ No (secuencial)",
                "ASR, Generación texto",
            ],
            "Transformer": [
                "Atención multi-cabeza (global)",
                "✅✅ Excelente (toda la secuencia)",
                "✅✅ Con paralelización",
                "Muchos",
                "✅ No aplica",
                "✅✅ Total",
                "LLMs: GPT, Llama, BERT...",
            ]
        }

        df_arch = pd.DataFrame(data)
        st.dataframe(df_arch.set_index("Característica"), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📉 Visualización: Degradación del Gradiente por Distancia")

        steps = list(range(1, 51))
        rnn_grad = [0.9**t for t in steps]
        lstm_grad = [max(0.85**t * 1.8, 0.05) for t in steps]
        gru_grad = [max(0.87**t * 1.7, 0.06) for t in steps]
        transformer_grad = [0.95 for _ in steps]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=rnn_grad, name="RNN", line=dict(color="#ef4444", width=2.5)))
        fig.add_trace(go.Scatter(x=steps, y=lstm_grad, name="LSTM", line=dict(color="#3b82f6", width=2.5)))
        fig.add_trace(go.Scatter(x=steps, y=gru_grad, name="GRU", line=dict(color="#10b981", width=2.5)))
        fig.add_trace(go.Scatter(x=steps, y=transformer_grad, name="Transformer", line=dict(color="#f59e0b", width=2.5, dash="dash")))
        fig.update_layout(
            title="Gradiente efectivo vs distancia temporal (simulación)",
            xaxis_title="Pasos hacia atrás en el tiempo",
            yaxis_title="Magnitud del gradiente (normalizada)",
            height=400,
            legend=dict(x=0.7, y=0.9)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Simulación ilustrativa — valores aproximados para fines educativos.")

        # Architecture diagrams in text
        st.markdown("### 🔬 Mecanismos de Compuerta")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🔴 RNN**
            ```
            h_t = tanh(W·[h_{t-1}, x_t] + b)
            ```
            - Un solo estado oculto
            - Sin mecanismo de olvido
            - Gradiente se multiplica repetidamente
            """)
        with col2:
            st.markdown("""
            **🔵 LSTM**
            ```
            f_t = σ(W_f·[h_{t-1}, x_t])  # forget
            i_t = σ(W_i·[h_{t-1}, x_t])  # input
            o_t = σ(W_o·[h_{t-1}, x_t])  # output
            C_t = f_t⊙C_{t-1} + i_t⊙C̃_t
            ```
            - Cell state como memoria de largo plazo
            - 3 compuertas con σ (0 a 1)
            """)
        with col3:
            st.markdown("""
            **🟢 GRU**
            ```
            z_t = σ(W_z·[h_{t-1}, x_t])  # update
            r_t = σ(W_r·[h_{t-1}, x_t])  # reset
            h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
            ```
            - Fusiona cell state y hidden state
            - 2 compuertas — más eficiente
            """)

    with tab3:
        st.markdown("## 🧮 HMM vs CRF — Modelos Probabilísticos de Secuencia")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### HMM (Hidden Markov Model)
            **Tipo:** Generativo — modela P(X, Y)
            
            **Supuestos:**
            - Markov: P(y_t | y_1..y_{t-1}) = P(y_t | y_{t-1})
            - Independencia: P(x_t | y_1..y_T) = P(x_t | y_t)
            
            **Parámetros:**
            - Probabilidades de transición: A[i,j] = P(y_t=j | y_{t-1}=i)
            - Probabilidades de emisión: B[j,x] = P(x_t=x | y_t=j)
            
            **Inferencia:** Algoritmo de Viterbi
            
            **Limitación en NER:** Solo puede "ver" la palabra actual para decidir la etiqueta
            """)

        with col2:
            st.markdown("""
            ### CRF (Conditional Random Field)
            **Tipo:** Discriminativo — modela P(Y | X)
            
            **Ventaja:** No asume independencia de observaciones
            
            **Features posibles:**
            - Palabra actual, anterior, siguiente
            - Prefijos/sufijos
            - Mayúsculas, dígitos, guiones
            - Etiquetas anteriores
            
            **Parámetros:** Pesos de features λ_k
            
            **Inferencia:** Programación dinámica sobre todo el CRF
            
            **Por qué supera a HMM:** Usa TODA la oración para decidir cada etiqueta
            """)

        st.markdown("---")
        st.markdown("### Ejemplo de NER: ¿Qué ve cada modelo?")

        example_sent = ["Juan", "Pérez", "trabaja", "en", "Google", "Colombia", "desde", "2020"]
        example_tags = ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O", "B-DATE"]

        st.markdown("**Oración:** `Juan Pérez trabaja en Google Colombia desde 2020`")

        hmm_features = ["Juan", "Pérez", "trabaja", "en", "Google", "Colombia", "desde", "2020"]
        crf_features = [
            "Juan + Mayúsc. + prev=START + next=Pérez",
            "Pérez + Mayúsc. + prev=Juan(B-PER) + next=trabaja",
            "trabaja + minúsc. + prev=Pérez(I-PER) + next=en",
            "en + minúsc. + prev=trabaja(O) + next=Google",
            "Google + Mayúsc. + prev=en(O) + next=Colombia",
            "Colombia + Mayúsc. + prev=Google(B-ORG) + next=desde",
            "desde + minúsc. + prev=Colombia(I-ORG) + next=2020",
            "2020 + dígitos + prev=desde(O)"
        ]

        df_ner = pd.DataFrame({
            "Token": example_sent,
            "Etiqueta": example_tags,
            "Features HMM": hmm_features,
            "Features CRF": crf_features
        })
        st.dataframe(df_ner, use_container_width=True, hide_index=True)

        st.success("✅ CRF usa el contexto completo (features ricas) vs HMM que solo usa la palabra actual → Mejor precisión en NER")
