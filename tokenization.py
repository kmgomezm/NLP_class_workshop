import streamlit as st
import re
from collections import Counter
import pandas as pd

def word_tokenize_simple(text):
    """Simple word-level tokenizer."""
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    return tokens

def character_tokenize(text):
    """Character-level tokenizer."""
    return list(text)

def bpe_simulate(text, vocab_size=50):
    """
    Simulated BPE tokenization (educational approximation).
    Real BPE requires a pre-trained vocab; this simulates the concept.
    """
    # Start with character-level tokens
    words = text.lower().split()
    tokens = []
    for word in words:
        chars = list(word)
        # Simulate some common BPE merges (bigrams)
        i = 0
        merged = []
        common_pairs = ["th", "he", "in", "er", "an", "re", "on", "en", "at", "es",
                       "ti", "or", "ar", "al", "te", "co", "de", "ra", "se", "nd",
                       "ing", "ion", "tion", "ed", "ly", "ment", "ness", "ful"]
        word_str = word
        result = []
        pos = 0
        while pos < len(word_str):
            matched = False
            for pair in sorted(common_pairs, key=len, reverse=True):
                if word_str[pos:pos+len(pair)] == pair:
                    result.append(f"[{pair}]")
                    pos += len(pair)
                    matched = True
                    break
            if not matched:
                result.append(word_str[pos])
                pos += 1
        tokens.extend(result)
        tokens.append("▁")  # word boundary marker
    return [t for t in tokens if t != "▁" or tokens.index(t) != len(tokens)-1]

def sentence_tokenize(text):
    """Simple sentence tokenizer."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def show():
    st.title("🔤 Tokenización & Encoding")
    st.markdown("Compara cómo diferentes estrategias de tokenización fragmentan el mismo texto.")
    st.markdown("---")

    default_text = "El procesamiento del lenguaje natural permite que las máquinas comprendan el texto humano. Las redes neuronales han revolucionado este campo."
    text = st.text_area(
        "✏️ Ingresa tu texto:",
        value=default_text,
        height=100
    )

    if not text.strip():
        st.warning("Por favor ingresa algún texto.")
        return

    st.markdown("---")
    st.markdown("## 🔬 Comparación de Tokenizadores")

    word_tokens = word_tokenize_simple(text)
    char_tokens = character_tokenize(text)
    bpe_tokens = bpe_simulate(text)
    sent_tokens = sentence_tokenize(text)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Word-level", f"{len(word_tokens)} tokens")
    with col2:
        st.metric("Char-level", f"{len(char_tokens)} tokens")
    with col3:
        st.metric("BPE (sim.)", f"{len(bpe_tokens)} tokens")
    with col4:
        st.metric("Sentence-level", f"{len(sent_tokens)} oraciones")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🔵 Word-level", "🔴 Char-level", "🟢 BPE (Llama)", "🟡 Sentence-level"])

    with tab1:
        st.markdown("### Tokenización a nivel de palabra")
        st.markdown("""
        **Cómo funciona:** Divide el texto por espacios y puntuación.  
        **Vocabulario:** Una entrada por palabra única.  
        **Problema:** Palabras fuera del vocabulario (OOV), no maneja morfología.
        """)
        cols = st.columns(6)
        for i, token in enumerate(word_tokens):
            cols[i % 6].markdown(
                f'<div style="background:#1e40af;color:white;padding:4px 8px;border-radius:4px;margin:2px;font-size:0.85em;text-align:center">{token}</div>',
                unsafe_allow_html=True
            )
        st.markdown(f"\n**Tokens únicos:** {len(set(word_tokens))} | **Total:** {len(word_tokens)}")

        # Frequency table
        freq = Counter(word_tokens)
        df_freq = pd.DataFrame(freq.most_common(10), columns=["Token", "Frecuencia"])
        st.markdown("**Top 10 tokens más frecuentes:**")
        st.dataframe(df_freq, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Tokenización a nivel de carácter")
        st.markdown("""
        **Cómo funciona:** Cada carácter es un token.  
        **Vocabulario:** Muy pequeño (~100 caracteres).  
        **Problema:** Secuencias muy largas, pierde información de palabras.
        """)
        # Show first 60 chars to avoid overflow
        display_chars = char_tokens[:80]
        cols = st.columns(8)
        for i, ch in enumerate(display_chars):
            label = "·" if ch == " " else ("↵" if ch == "\n" else ch)
            color = "#dc2626" if ch == " " else "#7c3aed"
            cols[i % 8].markdown(
                f'<div style="background:{color};color:white;padding:4px 6px;border-radius:4px;margin:2px;font-size:0.8em;text-align:center">{label}</div>',
                unsafe_allow_html=True
            )
        if len(char_tokens) > 80:
            st.markdown(f"*... y {len(char_tokens)-80} caracteres más*")

    with tab3:
        st.markdown("### BPE — Byte Pair Encoding (simulado)")
        st.markdown("""
        **Cómo funciona:** Fusiona iterativamente los pares de bytes/caracteres más frecuentes.  
        **Usado en:** GPT, Llama, RoBERTa, casi todos los LLMs modernos.  
        **Ventaja:** Balance entre vocabulario manejable y cobertura completa (no OOV).
        
        > ⚠️ *Esta es una simulación educativa del concepto BPE. El BPE real de Llama 3.3 usa un vocabulario de ~128,000 tokens entrenado en corpus masivos.*
        """)
        cols = st.columns(6)
        for i, token in enumerate(bpe_tokens[:48]):
            color = "#059669" if token.startswith("[") else "#0891b2"
            cols[i % 6].markdown(
                f'<div style="background:{color};color:white;padding:4px 8px;border-radius:4px;margin:2px;font-size:0.8em;text-align:center">{token}</div>',
                unsafe_allow_html=True
            )
        st.markdown("""
        **Comparativa de vocabularios:**
        | Modelo | Tamaño Vocabulario |
        |--------|-------------------|
        | BERT (WordPiece) | 30,522 |
        | GPT-2 (BPE) | 50,257 |
        | Llama 3.3 (BPE) | ~128,256 |
        | Word-level típico | 50,000-100,000 |
        """)

    with tab4:
        st.markdown("### Tokenización a nivel de oración")
        st.markdown("""
        **Cómo funciona:** Divide el texto en oraciones usando signos de puntuación.  
        **Uso:** Preprocesamiento para tareas que requieren análisis por oración.
        """)
        for i, sent in enumerate(sent_tokens):
            st.markdown(
                f'<div style="background:#f59e0b20;border-left:4px solid #f59e0b;padding:8px 12px;margin:4px 0;border-radius:0 4px 4px 0">'
                f'<strong>Oración {i+1}:</strong> {sent}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("## 📊 One-Hot Encoding")
    st.markdown("Visualiza la representación One-Hot de las primeras palabras del texto.")

    vocab = sorted(set(word_tokens))
    if len(vocab) > 0:
        display_words = word_tokens[:min(8, len(word_tokens))]
        display_vocab = vocab[:min(15, len(vocab))]

        matrix_data = []
        for word in display_words:
            row = [1 if word == v else 0 for v in display_vocab]
            matrix_data.append(row)

        df_onehot = pd.DataFrame(matrix_data, columns=display_vocab, index=display_words)

        # Style the dataframe
        def highlight_one(val):
            if val == 1:
                return "background-color: #1e40af; color: white; font-weight: bold"
            return "background-color: #f1f5f9; color: #94a3b8"

        st.dataframe(df_onehot.style.applymap(highlight_one), use_container_width=True)
        st.caption(f"Dimensionalidad: {len(display_words)} palabras × {len(vocab)} dimensiones del vocabulario completo")

        st.info(f"""
        💡 **Observa la dispersión:** En un vocabulario de {len(vocab)} palabras, cada token tiene 
        solo **1 valor en 1** y el resto en **0**. Con vocabularios reales de 50,000+ palabras, 
        esto genera vectores extremadamente dispersos e ineficientes.
        """)
