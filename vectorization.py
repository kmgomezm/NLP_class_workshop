import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

def simple_tokenize(text):
    stopwords_es = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
                    "en", "y", "a", "que", "se", "por", "con", "para", "es", "son",
                    "al", "lo", "le", "su", "sus", "no", "si", "más", "pero", "o"}
    tokens = re.findall(r'\b[a-záéíóúüñ]+\b', text.lower())
    return [t for t in tokens if t not in stopwords_es and len(t) > 2]

def build_bow(corpus):
    tokenized = [simple_tokenize(doc) for doc in corpus]
    vocab = sorted(set(word for doc in tokenized for word in doc))
    matrix = []
    for tokens in tokenized:
        freq = Counter(tokens)
        row = [freq.get(word, 0) for word in vocab]
        matrix.append(row)
    return matrix, vocab, tokenized

def compute_tfidf(corpus):
    tokenized = [simple_tokenize(doc) for doc in corpus]
    vocab = sorted(set(word for doc in tokenized for word in doc))
    N = len(corpus)

    # TF
    tf_matrix = []
    for tokens in tokenized:
        freq = Counter(tokens)
        total = len(tokens) if tokens else 1
        tf_matrix.append({w: freq.get(w, 0) / total for w in vocab})

    # IDF
    idf = {}
    for word in vocab:
        df = sum(1 for tokens in tokenized if word in tokens)
        idf[word] = np.log((N + 1) / (df + 1)) + 1  # smoothed

    # TF-IDF
    tfidf_matrix = []
    for tf_doc in tf_matrix:
        row = [tf_doc[w] * idf[w] for w in vocab]
        tfidf_matrix.append(row)

    # Normalize
    tfidf_matrix_norm = []
    for row in tfidf_matrix:
        norm = np.sqrt(sum(v**2 for v in row))
        tfidf_matrix_norm.append([v / norm if norm > 0 else 0 for v in row])

    return tfidf_matrix_norm, vocab, idf

DEFAULT_CORPUS = [
    "El procesamiento de lenguaje natural usa algoritmos para entender texto",
    "Las redes neuronales aprenden representaciones del lenguaje automáticamente",
    "Los modelos de lenguaje grandes como GPT generan texto coherente",
    "El aprendizaje profundo ha mejorado el reconocimiento de entidades nombradas",
    "Los transformers revolucionaron el procesamiento del lenguaje con la atención"
]

def show():
    st.title("📊 Vectorización Clásica")
    st.markdown("Explora las representaciones BoW y TF-IDF sobre un corpus personalizable.")
    st.markdown("---")

    st.markdown("### 📝 Define tu Corpus")
    st.markdown("Ingresa cada documento en una línea separada:")

    corpus_text = st.text_area(
        "Corpus de documentos:",
        value="\n".join(DEFAULT_CORPUS),
        height=160
    )
    corpus = [line.strip() for line in corpus_text.strip().split("\n") if line.strip()]

    if len(corpus) < 2:
        st.warning("Ingresa al menos 2 documentos para visualizar la vectorización.")
        return

    doc_labels = [f"Doc {i+1}" for i in range(len(corpus))]

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🟦 Bag of Words (BoW)", "🟩 TF-IDF", "📐 Comparativa"])

    with tab1:
        st.markdown("## 🟦 Bag of Words")
        st.markdown("""
        **BoW** transforma cada documento en un vector de frecuencias de palabras.
        Ignora el orden pero captura la presencia de términos.
        """)
        bow_matrix, vocab_bow, _ = build_bow(corpus)
        df_bow = pd.DataFrame(bow_matrix, columns=vocab_bow, index=doc_labels)

        # Show heatmap
        fig = px.imshow(
            df_bow.values,
            labels=dict(x="Términos", y="Documentos", color="Frecuencia"),
            x=vocab_bow,
            y=doc_labels,
            color_continuous_scale="Blues",
            title="Matriz BoW — Frecuencia de términos"
        )
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Matriz BoW completa:**")
        st.dataframe(df_bow.style.background_gradient(cmap="Blues"), use_container_width=True)

        # Top terms
        total_freq = df_bow.sum().sort_values(ascending=False).head(15)
        fig_bar = px.bar(
            x=total_freq.index,
            y=total_freq.values,
            title="Top 15 términos más frecuentes en el corpus",
            labels={"x": "Término", "y": "Frecuencia total"},
            color=total_freq.values,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.warning("""
        ⚠️ **Limitaciones de BoW:**
        - No considera el orden de las palabras
        - Favorece términos frecuentes globalmente (ej. stopwords)
        - Documentos más largos tienen vectores con valores más altos
        """)

    with tab2:
        st.markdown("## 🟩 TF-IDF")
        st.markdown("""
        **TF-IDF** (Term Frequency–Inverse Document Frequency) pondera los términos según:
        - **TF**: Qué tan frecuente es una palabra en un documento
        - **IDF**: Qué tan *rara* es la palabra en todo el corpus
        
        `TF-IDF(t, d) = TF(t, d) × IDF(t)`
        """)

        tfidf_matrix, vocab_tfidf, idf_scores = compute_tfidf(corpus)
        df_tfidf = pd.DataFrame(tfidf_matrix, columns=vocab_tfidf, index=doc_labels)

        fig = px.imshow(
            df_tfidf.values,
            labels=dict(x="Términos", y="Documentos", color="TF-IDF"),
            x=vocab_tfidf,
            y=doc_labels,
            color_continuous_scale="Greens",
            title="Matriz TF-IDF (normalizada)"
        )
        fig.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Matriz TF-IDF completa:**")
        st.dataframe(
            df_tfidf.round(4).style.background_gradient(cmap="Greens"),
            use_container_width=True
        )

        # IDF scores
        df_idf = pd.DataFrame(
            sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)[:20],
            columns=["Término", "IDF Score"]
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Términos con mayor IDF (más raros/discriminativos):**")
            st.dataframe(df_idf.head(10), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Términos con menor IDF (más comunes):**")
            st.dataframe(df_idf.tail(10), hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("## 📐 BoW vs TF-IDF — Comparativa")
        st.markdown("""
        Observa cómo TF-IDF re-pondera los términos respecto a BoW puro.
        Los términos que aparecen en muchos documentos reciben menor peso en TF-IDF.
        """)

        bow_matrix_n, _, _ = build_bow(corpus)
        bow_arr = np.array(bow_matrix_n, dtype=float)
        # Normalize BoW for comparison
        norms = np.linalg.norm(bow_arr, axis=1, keepdims=True)
        bow_norm = np.divide(bow_arr, norms, where=norms != 0)
        tfidf_arr = np.array(tfidf_matrix)

        common_vocab = vocab_tfidf[:min(12, len(vocab_tfidf))]
        common_idx = [vocab_tfidf.index(w) for w in common_vocab]

        doc_sel = st.selectbox("Selecciona un documento para comparar:", doc_labels)
        doc_idx = doc_labels.index(doc_sel)

        bow_vals = [bow_norm[doc_idx][common_idx[i]] for i in range(len(common_vocab))]
        tfidf_vals = [tfidf_arr[doc_idx][common_idx[i]] for i in range(len(common_vocab))]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="BoW (norm.)", x=common_vocab, y=bow_vals, marker_color="#3b82f6"))
        fig_comp.add_trace(go.Bar(name="TF-IDF", x=common_vocab, y=tfidf_vals, marker_color="#10b981"))
        fig_comp.update_layout(
            barmode="group",
            title=f"BoW vs TF-IDF — {doc_sel}",
            xaxis_tickangle=-30,
            height=400
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### BoW
            | Ventaja | Desventaja |
            |---------|------------|
            | Simple e intuitivo | No discrimina términos frecuentes |
            | Rápido de calcular | Favorece documentos largos |
            | Bueno para documentos similares | Sin semántica ni orden |
            """)
        with col2:
            st.markdown("""
            ### TF-IDF
            | Ventaja | Desventaja |
            |---------|------------|
            | Penaliza términos muy comunes | Aún sin semántica |
            | Identifica términos clave | No captura sinónimos |
            | Estándar en recuperación de información | Sensible al tamaño del corpus |
            """)
