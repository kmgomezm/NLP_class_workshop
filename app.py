import streamlit as st
import re
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(page_title="Taller NLP & LLMs - EAFIT", page_icon="🧠", layout="wide")

# ── Helpers ────────────────────────────────────────────────────────────────────

def tokenize(text):
    sw = {"el","la","los","las","un","una","de","del","en","y","a","que","se","por","con","es","son","al","no","su","sus"}
    return [t for t in re.findall(r'\b[a-záéíóúüñ]+\b', text.lower()) if t not in sw and len(t) > 2]

def build_bow(corpus):
    tok = [tokenize(d) for d in corpus]
    vocab = sorted(set(w for d in tok for w in d))
    return [[Counter(d).get(w, 0) for w in vocab] for d in tok], vocab

def build_tfidf(corpus):
    tok = [tokenize(d) for d in corpus]
    vocab = sorted(set(w for d in tok for w in d))
    N = len(corpus)
    idf = {w: np.log((N+1)/(sum(1 for d in tok if w in d)+1))+1 for w in vocab}
    matrix = []
    for d in tok:
        total = len(d) or 1
        tf = Counter(d)
        row = [(tf.get(w,0)/total)*idf[w] for w in vocab]
        norm = np.linalg.norm(row) or 1
        matrix.append([v/norm for v in row])
    return matrix, vocab

def softmax(logits, T):
    T = max(T, 0.01)
    e = np.exp((np.array(logits) - max(logits)) / T)
    return e / e.sum()

def groq_call(api_key, messages, system=None, model="llama-3.3-70b-versatile",
              temperature=0.7, top_p=0.9, max_tokens=600):
    from groq import Groq
    msgs = ([{"role":"system","content":system}] if system else []) + messages
    t0 = time.time()
    r = Groq(api_key=api_key).chat.completions.create(
        model=model, messages=msgs,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    return r.choices[0].message.content, time.time()-t0, r.usage

def render_tokens(tokens, color, max_n=60):
    html = "".join(
        f'<span style="background:{color};color:white;padding:3px 7px;border-radius:4px;'
        f'margin:2px;display:inline-block;font-size:.85em">{t if str(t).strip() else "·"}</span>'
        for t in tokens[:max_n]
    )
    if len(tokens) > max_n:
        html += f"<span style='color:gray'> ...+{len(tokens)-max_n} más</span>"
    st.markdown(html, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("🧠 Taller NLP & LLMs")
st.sidebar.caption("Maestría en Ciencia de Datos · EAFIT 2026-1")
st.sidebar.divider()
page = st.sidebar.radio("Ir a:", [
    "🏠 Inicio", "📝 Quiz", "🔤 Tokenización",
    "📊 Vectorización", "🔗 Secuencias", "🤖 Lab LLM", "💬 Agente"
])
st.sidebar.divider()
api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
model   = st.sidebar.selectbox("Modelo Groq", ["llama-3.3-70b-versatile","llama3-8b-8192"])

# ══════════════════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.title("🧠 Taller NLP & LLMs — EAFIT")
    st.markdown("**Docente:** Jorge Iván Padilla-Buriticá &nbsp;|&nbsp; **Curso:** IA ECA&I Posgrado &nbsp;|&nbsp; **Periodo:** 2026-1")
    st.divider()
    st.markdown("""
    ## Objetivo
    Explorar la evolución del NLP desde representaciones clásicas hasta agentes conversacionales con **Llama 3.3 vía Groq**.

    | Sección | Contenido |
    |---------|-----------|
    | 📝 Quiz | 4 preguntas conceptuales con retroalimentación |
    | 🔤 Tokenización | Word-level, BPE, BERT (WordPiece), GPT-2, Llama — comparativa real |
    | 📊 Vectorización | Matrices BoW y TF-IDF con corpus personalizable |
    | 🔗 Secuencias | N-grams, comparativa RNN/LSTM/GRU/Transformer |
    | 🤖 Lab LLM | Playground de temperatura + visualización Softmax |
    | 💬 Agente | Agente especializado con métricas: latencia, TPS, score |

    ## Configuración
    Ingresa tu **Groq API Key** gratuita ([console.groq.com](https://console.groq.com)) en la barra lateral.
    """)
    st.info('"El lenguaje es el vestido del pensamiento." — Samuel Johnson')

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Quiz":
    st.title("📝 Quiz Conceptual")

    QUIZ = [
        {
            "q": "¿Cuál es la diferencia fundamental entre One-Hot Encoding y Embeddings densos?",
            "opts": {
                "A": "One-Hot captura semántica; los Embeddings no.",
                "B": "One-Hot genera vectores dispersos de alta dimensión sin semántica; los Embeddings son densos y capturan relaciones semánticas.",
                "C": "Ambos tienen la misma dimensionalidad.",
                "D": "Los Embeddings siempre requieren más memoria que One-Hot."
            },
            "ans": "B",
            "exp": "**One-Hot**: dimensión = tamaño del vocabulario, vectores dispersos, sin semántica. **Embeddings**: 100-300 dims, densos, capturan `rey - hombre + mujer ≈ reina`."
        },
        {
            "q": "¿Por qué un CRF suele superar a un HMM en NER?",
            "opts": {
                "A": "Porque CRF es más rápido de entrenar.",
                "B": "Porque HMM no puede manejar texto.",
                "C": "CRF es discriminativo y condiciona toda la secuencia; HMM es generativo y asume independencia condicional entre observaciones.",
                "D": "CRF requiere menos datos."
            },
            "ans": "C",
            "exp": "**HMM** modela P(X,Y) asumiendo independencia entre observaciones. **CRF** modela P(Y|X) sin ese supuesto, usando features arbitrarias (prefijos, vecinos, capitalización)."
        },
        {
            "q": "¿Cómo mitigan LSTM y GRU el vanishing gradient?",
            "opts": {
                "A": "Mediante compuertas que regulan el flujo de gradiente a través del tiempo.",
                "B": "Usando ReLU en lugar de tanh.",
                "C": "Aumentando el batch size.",
                "D": "Reduciendo capas."
            },
            "ans": "A",
            "exp": "**LSTM**: forget/input/output gates + cell state como autopista del gradiente. **GRU**: update/reset gates. Ambos evitan la multiplicación repetida de valores < 1."
        },
        {
            "q": "Con temperatura T ≈ 2.0 en un LLM, ¿qué le ocurre al Softmax?",
            "opts": {
                "A": "El modelo siempre elige el token más probable.",
                "B": "Las respuestas se acortan.",
                "C": "No hay efecto.",
                "D": "La distribución se aplana: todos los tokens tienen probabilidades similares → máxima aleatoriedad."
            },
            "ans": "D",
            "exp": "`P(i) = exp(logit_i / T) / Σ exp(logit_j / T)`. Con T→0: greedy. T=1: distribución original. T→2: uniforme → caótico."
        }
    ]

    if "qa" not in st.session_state: st.session_state.qa = {}
    if "qsubmit" not in st.session_state: st.session_state.qsubmit = False

    for i, q in enumerate(QUIZ):
        with st.expander(f"**P{i+1}.** {q['q']}", expanded=True):
            r = st.radio("", list(q["opts"].keys()),
                         format_func=lambda x, q=q: f"{x}) {q['opts'][x]}",
                         key=f"q{i}", index=None)
            if r: st.session_state.qa[i] = r

    col1, col2 = st.columns(2)
    if col1.button("✅ Verificar", type="primary"): st.session_state.qsubmit = True
    if col2.button("🔄 Reiniciar"):
        st.session_state.qa = {}; st.session_state.qsubmit = False; st.rerun()

    if st.session_state.qsubmit:
        score = sum(1 for i, q in enumerate(QUIZ) if st.session_state.qa.get(i) == q["ans"])
        st.divider()
        [st.error, st.warning, st.warning, st.success, st.success][score](f"**{score}/{len(QUIZ)} correctas**")
        for i, q in enumerate(QUIZ):
            ua = st.session_state.qa.get(i)
            ok = ua == q["ans"]
            with st.expander(f"{'✅' if ok else '❌'} P{i+1}"):
                if not ok: st.error(f"Tu respuesta: {ua} · Correcta: **{q['ans']}**")
                else: st.success(f"Correcto: **{q['ans']}**")
                st.markdown(q["exp"])

# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔤 Tokenización":
    st.title("🔤 Tokenización & Encoding")

    text = st.text_area(
        "Texto:",
        value="El procesamiento del lenguaje natural permite que las máquinas comprendan el texto humano.",
        height=90
    )
    if not text.strip():
        st.stop()

    # ── Tokenizadores locales ─────────────────────────────────────────────────
    word_tok = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    char_tok  = list(text)

    # BPE simulado
    common = ["ción","ment","ness","ing","tion","er","al","es","en","or","ar","el","la","os","as"]
    def bpe_sim(t):
        out = []
        for w in t.lower().split():
            pos, res = 0, []
            while pos < len(w):
                m = next((p for p in sorted(common, key=len, reverse=True) if w[pos:].startswith(p)), None)
                if m: res.append(f"[{m}]"); pos += len(m)
                else: res.append(w[pos]); pos += 1
            out += res
        return out
    bpe_tok = bpe_sim(text)

    # ── GPT-2 via tiktoken ────────────────────────────────────────────────────
    def get_gpt2_tokens(text):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            ids = enc.encode(text)
            return [enc.decode([i]) for i in ids], ids
        except Exception as e:
            return None, str(e)

    # ── BERT WordPiece via transformers ───────────────────────────────────────
    def get_bert_tokens(text):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            tokens = tok.tokenize(text)
            ids    = tok.encode(text)
            return tokens, list(ids)
        except Exception as e:
            return None, str(e)

    # ── Llama/GPT-4 cl100k via tiktoken ──────────────────────────────────────
    def get_llama_tokens(text):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(text)
            return [enc.decode([i]) for i in ids], ids
        except Exception as e:
            return None, str(e)

    gpt2_tok,  gpt2_ids  = get_gpt2_tokens(text)
    bert_tok,  bert_ids  = get_bert_tokens(text)
    llama_tok, llama_ids = get_llama_tokens(text)

    # ── Métricas resumen ──────────────────────────────────────────────────────
    st.divider()
    cols = st.columns(5)
    cols[0].metric("Word-level",  len(word_tok))
    cols[1].metric("Char-level",  len(char_tok))
    cols[2].metric("GPT-2 (BPE)", len(gpt2_tok)  if gpt2_tok  else 0)
    cols[3].metric("BERT (WP)",   len(bert_tok)  if bert_tok  else 0)
    cols[4].metric("Llama/GPT-4", len(llama_tok) if llama_tok else 0)

    st.divider()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Word-level", "Char-level", "BPE simulado", "GPT-2", "BERT", "Llama / GPT-4"
    ])

    with tab1:
        st.markdown("Divide por espacios y puntuación. **Vocabulario** = palabras únicas.")
        render_tokens(word_tok, "#1d4ed8")

    with tab2:
        st.markdown("Cada carácter es un token. Vocabulario ~100, secuencias muy largas.")
        render_tokens(char_tok, "#7c3aed", 80)

    with tab3:
        st.markdown("Fusiona pares frecuentes iterativamente. **Simulación educativa** del algoritmo BPE.")
        render_tokens(bpe_tok, "#047857")

    with tab4:
        st.markdown("""
        ### GPT-2 — BPE real (`tiktoken`)
        - **Vocabulario:** 50,257 tokens  
        - **Algoritmo:** Byte Pair Encoding sobre bytes UTF-8  
        - **Usado en:** GPT-2, GPT-3
        """)
        if gpt2_tok:
            render_tokens(gpt2_tok, "#b45309")
            st.divider()
            st.dataframe(pd.DataFrame({"Token": gpt2_tok, "ID": gpt2_ids}),
                         use_container_width=True, hide_index=True)
        else:
            st.warning(f"tiktoken no disponible: {gpt2_ids}")

    with tab5:
        st.markdown("""
        ### BERT — WordPiece real (`transformers`)
        - **Vocabulario:** 119,547 tokens (multilingual-cased)  
        - **Algoritmo:** WordPiece — divide palabras desconocidas con prefijo `##`  
        - **Tokens especiales:** `[CLS]` al inicio, `[SEP]` al final
        """)
        if bert_tok:
            render_tokens(bert_tok, "#9f1239")
            st.divider()
            # bert_ids includes [CLS] and [SEP], bert_tok does not — align by trimming
            aligned_ids = bert_ids[1:len(bert_tok)+1]
            st.dataframe(pd.DataFrame({"Token": bert_tok, "ID": aligned_ids}),
                         use_container_width=True, hide_index=True)
            st.info("El prefijo `##` indica continuación de una palabra dividida por WordPiece.")
        else:
            st.warning(f"transformers no disponible: {bert_ids}")

    with tab6:
        st.markdown("""
        ### Llama 3 / GPT-4 — BPE `cl100k_base` (`tiktoken`)
        - **Vocabulario:** 100,277 tokens  
        - **Algoritmo:** BPE sobre bytes UTF-8 (misma base que GPT-4 y Llama 3)  
        - **Mejora vs GPT-2:** Mejor cobertura multilingüe y de código
        """)
        if llama_tok:
            render_tokens(llama_tok, "#065f46")
            st.divider()
            st.dataframe(pd.DataFrame({"Token": llama_tok, "ID": llama_ids}),
                         use_container_width=True, hide_index=True)
        else:
            st.warning(f"tiktoken no disponible: {llama_ids}")

    # ── Comparativa visual ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Comparativa de tokenizadores")

    nombres   = ["Word-level", "Char-level", "BPE sim.", "GPT-2", "BERT (WP)", "Llama/GPT-4"]
    n_tokens  = [
        len(word_tok), len(char_tok), len(bpe_tok),
        len(gpt2_tok)  if gpt2_tok  else 0,
        len(bert_tok)  if bert_tok  else 0,
        len(llama_tok) if llama_tok else 0,
    ]
    vocabs    = ["~corpus", "~100", "~sim.", "50,257", "119,547", "100,277"]
    algos     = ["Whitespace", "Carácter", "BPE sim.", "BPE bytes", "WordPiece", "BPE bytes"]
    modelos_r = ["Clásico", "Clásico", "Educativo", "GPT-2/3", "BERT", "GPT-4/Llama 3"]
    colors    = ["#1d4ed8","#7c3aed","#047857","#b45309","#9f1239","#065f46"]

    fig = go.Figure(go.Bar(
        x=nombres, y=n_tokens, marker_color=colors,
        text=n_tokens, textposition="outside"
    ))
    fig.update_layout(title="# Tokens generados por tokenizador", yaxis_title="# Tokens", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "Tokenizador": nombres,
        "# Tokens":    n_tokens,
        "Vocab size":  vocabs,
        "Algoritmo":   algos,
        "Modelo":      modelos_r,
    }), hide_index=True, use_container_width=True)

    # ── One-Hot ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔢 One-Hot Encoding (word-level)")
    vocab  = sorted(set(word_tok))
    sample = word_tok[:8]
    df_oh  = pd.DataFrame(
        [[1 if w == v else 0 for v in vocab[:14]] for w in sample],
        index=sample, columns=vocab[:14]
    )
    st.dataframe(df_oh.style.applymap(
        lambda v: "background:#1d4ed8;color:white" if v == 1 else "background:#f1f5f9;color:#94a3b8"
    ))
    st.caption(f"Dimensión real: {len(sample)} × {len(vocab)} — vectores {100*1/len(vocab):.1f}% densos")

# ══════════════════════════════════════════════════════════════════════════════
# VECTORIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Vectorización":
    st.title("📊 Vectorización Clásica — BoW & TF-IDF")

    DEFAULT = "\n".join([
        "El procesamiento de lenguaje natural usa algoritmos para entender texto",
        "Las redes neuronales aprenden representaciones del lenguaje automáticamente",
        "Los modelos de lenguaje grandes como GPT generan texto coherente",
        "El aprendizaje profundo mejoró el reconocimiento de entidades nombradas",
        "Los transformers revolucionaron el NLP con mecanismos de atención"
    ])
    raw    = st.text_area("Corpus (un documento por línea):", value=DEFAULT, height=140)
    corpus = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if len(corpus) < 2:
        st.warning("Ingresa al menos 2 documentos."); st.stop()

    labels = [f"Doc {i+1}" for i in range(len(corpus))]
    tab1, tab2, tab3 = st.tabs(["BoW", "TF-IDF", "Comparativa"])

    with tab1:
        bow, vocab = build_bow(corpus)
        df = pd.DataFrame(bow, columns=vocab, index=labels)
        st.plotly_chart(px.imshow(df.values, x=vocab, y=labels, color_continuous_scale="Blues",
                                   title="Matriz BoW", labels=dict(color="Frecuencia")), use_container_width=True)
        top = pd.Series(df.values.sum(axis=0), index=vocab).nlargest(15)
        st.plotly_chart(px.bar(x=top.index, y=top.values, title="Top 15 términos",
                                labels={"x":"Término","y":"Frecuencia total"},
                                color=top.values, color_continuous_scale="Blues"), use_container_width=True)

    with tab2:
        tfidf, vocab_t = build_tfidf(corpus)
        df_t = pd.DataFrame(tfidf, columns=vocab_t, index=labels)
        st.plotly_chart(px.imshow(df_t.values, x=vocab_t, y=labels, color_continuous_scale="Greens",
                                   title="Matriz TF-IDF (normalizada)", labels=dict(color="Score")), use_container_width=True)

    with tab3:
        doc = st.selectbox("Documento:", labels)
        idx = labels.index(doc)
        bow2, v2    = build_bow(corpus)
        tfidf2, vt2 = build_tfidf(corpus)
        common_v    = [w for w in v2 if w in vt2][:14]
        bow_n       = np.array(bow2[idx], dtype=float)
        bow_n       = bow_n / (bow_n.max() or 1)
        bow_vals    = [bow_n[v2.index(w)]    for w in common_v]
        tfidf_vals  = [tfidf2[idx][vt2.index(w)] for w in common_v]
        fig = go.Figure([
            go.Bar(name="BoW (norm.)", x=common_v, y=bow_vals,   marker_color="#3b82f6"),
            go.Bar(name="TF-IDF",     x=common_v, y=tfidf_vals, marker_color="#10b981")
        ])
        fig.update_layout(barmode="group", title=f"BoW vs TF-IDF — {doc}")
        st.plotly_chart(fig, use_container_width=True)
        st.info("TF-IDF reduce el peso de términos muy frecuentes, destacando los términos discriminativos de cada documento.")

# ══════════════════════════════════════════════════════════════════════════════
# SECUENCIAS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Secuencias":
    st.title("🔗 Modelado de Secuencias")

    tab1, tab2, tab3 = st.tabs(["N-grams", "RNN / LSTM / GRU", "HMM vs CRF"])

    with tab1:
        text = st.text_area("Texto:", value="El aprendizaje profundo ha transformado el procesamiento del lenguaje natural. Los modelos neuronales aprenden representaciones del lenguaje de manera automática. El lenguaje natural presenta desafíos únicos para los sistemas de inteligencia artificial.", height=110)
        tokens = tokenize(text)
        if len(tokens) < 3:
            st.warning("Ingresa más texto."); st.stop()

        n = st.slider("N:", 2, 5, 2)
        k = st.slider("Top K:", 5, 25, 12)

        ngrams    = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        top       = Counter(ngrams).most_common(k)
        labels_ng = [" ".join(g) for g, _ in top]
        counts_ng = [c for _, c in top]

        st.plotly_chart(px.bar(x=counts_ng[::-1], y=labels_ng[::-1], orientation="h",
                                title=f"Top {k} {n}-grams", labels={"x":"Frecuencia","y":""},
                                color=counts_ng[::-1], color_continuous_scale="Viridis"), use_container_width=True)

        rows = []
        for ni in range(1, 6):
            ngs = [tuple(tokens[i:i+ni]) for i in range(len(tokens)-ni+1)]
            u = len(set(ngs)); t = len(ngs)
            rows.append({"N": ni, "Total": t, "Únicos": u, "Ratio": round(u/t, 3) if t else 0})
        df_cov = pd.DataFrame(rows)
        fig2 = go.Figure([
            go.Bar(name="Total",  x=df_cov.N, y=df_cov.Total,  marker_color="#3b82f6"),
            go.Bar(name="Únicos", x=df_cov.N, y=df_cov.Únicos, marker_color="#10b981")
        ])
        fig2.update_layout(barmode="group", title="Total vs Únicos por N", xaxis_title="N")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### Comparativa de arquitecturas")
        df_arch = pd.DataFrame({
            "": ["Memoria","Deps. largas","Vanishing grad.","Paralelizable","Uso típico"],
            "RNN":         ["Estado h_t","❌ Pobre","❌ Severo","❌","Demos, tareas simples"],
            "LSTM":        ["Cell + hidden","✅ Bueno","✅ Mitigado","❌","NER, traducción"],
            "GRU":         ["Hidden (gates)","✅ Bueno","✅ Mitigado","❌","ASR, generación"],
            "Transformer": ["Atención global","✅✅ Excelente","✅ No aplica","✅✅","GPT, Llama, BERT"],
        }).set_index("")
        st.dataframe(df_arch, use_container_width=True)

        steps = list(range(1, 51))
        fig = go.Figure([
            go.Scatter(x=steps, y=[0.9**t for t in steps],                  name="RNN",          line=dict(color="#ef4444")),
            go.Scatter(x=steps, y=[max(0.92**t*1.5, 0.05) for t in steps],  name="LSTM",         line=dict(color="#3b82f6")),
            go.Scatter(x=steps, y=[max(0.93**t*1.4, 0.06) for t in steps],  name="GRU",          line=dict(color="#10b981")),
            go.Scatter(x=steps, y=[0.95]*50,                                 name="Transformer",  line=dict(color="#f59e0b", dash="dash")),
        ])
        fig.update_layout(title="Gradiente efectivo vs distancia temporal (simulación)",
                          xaxis_title="Pasos hacia atrás", yaxis_title="Magnitud gradiente", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Simulación ilustrativa — valores aproximados.")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### HMM — Generativo
            Modela **P(X, Y)**. Asume:
            - Markov: estado depende solo del anterior
            - Independencia: observación depende solo del estado actual

            **Features:** solo la palabra actual
            """)
        with col2:
            st.markdown("""
            ### CRF — Discriminativo
            Modela **P(Y | X)**. Sin supuesto de independencia.

            **Features:** palabra actual/anterior/siguiente, mayúsculas, prefijos, sufijos, etiquetas vecinas
            """)
        st.success("**¿Por qué CRF gana en NER?** Ve toda la oración para decidir cada etiqueta, mientras HMM solo ve la palabra actual.")
        st.dataframe(pd.DataFrame({
            "Token":    ["Juan","Pérez","trabaja","en","Google","Colombia"],
            "Etiqueta": ["B-PER","I-PER","O","O","B-ORG","I-ORG"],
            "HMM ve":   ["Juan","Pérez","trabaja","en","Google","Colombia"],
            "CRF ve":   ["Juan+May+prev=START","Pérez+May+prev=B-PER","minúsc+prev=I-PER","minúsc","Google+May+prev=O","Colombia+May+prev=B-ORG"],
        }), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# LAB LLM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Lab LLM":
    st.title("🤖 Laboratorio LLM — Temperatura & Top-p")

    tab1, tab2 = st.tabs(["Playground", "Visualización Softmax"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            temp    = st.slider("🌡️ Temperatura:", 0.0, 2.0, 0.7, 0.1)
            top_p   = st.slider("🎯 Top-p:", 0.01, 1.0, 0.9, 0.01)
            max_tok = st.slider("📏 Max tokens:", 50, 600, 250, 50)
            label   = {temp < 0.3: "Determinista, preciso",
                       0.3 <= temp < 1.0: "Balanceado",
                       1.0 <= temp < 1.5: "Creativo",
                       temp >= 1.5: "Muy aleatorio"}[True]
            st.caption(f"→ {label}")
        with c2:
            prompts = {
                "Explicación técnica": "Explica brevemente qué es un embedding en NLP.",
                "Historia":            "Escribe el inicio de una historia de ciencia ficción sobre IA.",
                "Poema":               "Escribe un poema corto sobre el aprendizaje automático.",
                "Personalizado":       ""
            }
            tipo   = st.selectbox("Tipo:", list(prompts.keys()))
            prompt = st.text_area("Prompt:", value=prompts[tipo], height=90)

        if st.button("🚀 Generar", type="primary"):
            if not api_key:
                st.error("Ingresa tu API Key en la barra lateral.")
            elif not prompt.strip():
                st.warning("Escribe un prompt.")
            else:
                with st.spinner("Generando..."):
                    try:
                        content, lat, usage = groq_call(
                            api_key, [{"role":"user","content":prompt}],
                            model=model, temperature=temp, top_p=top_p, max_tokens=max_tok
                        )
                        st.markdown(content)
                        st.divider()
                        m1, m2, m3 = st.columns(3)
                        tps = usage.total_tokens / lat if lat > 0 else 0
                        m1.metric("⏱️ Latencia", f"{lat:.2f}s")
                        m2.metric("⚡ TPS",      f"{tps:.0f}")
                        m3.metric("🪙 Tokens",   usage.total_tokens)
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab2:
        st.markdown("**Fórmula:** `P(i) = exp(logit_i / T) / Σ exp(logit_j / T)`")
        T = st.slider("🌡️ T:", 0.1, 2.0, 1.0, 0.1, key="sv")
        tokens_eg = ["el","la","un","de","en","que","y","los","se","por"]
        logits    = [3.2, 2.8, 2.1, 1.9, 1.5, 1.2, 0.9, 0.7, 0.4, 0.1]
        p1 = softmax(logits, 1.0)
        pt = softmax(logits, T)
        fig = go.Figure([
            go.Bar(name="T=1.0",  x=tokens_eg, y=p1, marker_color="#3b82f6", opacity=0.6),
            go.Bar(name=f"T={T}", x=tokens_eg, y=pt, marker_color="#ef4444")
        ])
        fig.update_layout(barmode="group", title=f"Distribución con T=1.0 vs T={T}", height=380)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Entropía T=1",    f"{-np.sum(p1*np.log(p1+1e-10)):.3f}")
        c2.metric(f"Entropía T={T}", f"{-np.sum(pt*np.log(pt+1e-10)):.3f}")
        c3.metric("P(top token)",    f"{pt.max():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# AGENTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Agente":
    st.title("💬 Agente Conversacional")

    AGENTS = {
        "🎓 Consultor NLP/ML":    "Eres ARIA, consultora académica experta en NLP y ML para estudiantes de Maestría en Ciencia de Datos de EAFIT. Responde de forma clara y didáctica con ejemplos.",
        "💻 Asistente Python":    "Eres un experto en Python y librerías NLP (Hugging Face, scikit-learn, NLTK, Groq). Da código limpio, comentado y funcional.",
        "⚽ Experto Deportivo":   "Eres un analista deportivo experto en fútbol colombiano e internacional. Das análisis precisos y con datos.",
        "🧘 Coach Productividad": "Eres un coach de productividad para estudiantes de posgrado. Das consejos basados en evidencia sobre gestión del tiempo y bienestar."
    }

    agent     = st.selectbox("Agente:", list(AGENTS.keys()))
    temp_a    = st.sidebar.slider("🌡️ Temperatura agente:", 0.0, 1.5, 0.7, 0.1, key="at")
    auto_eval = st.sidebar.checkbox("🔍 LLM-as-Judge", value=True)

    if "agent" not in st.session_state or st.session_state.agent != agent:
        st.session_state.agent = agent
        st.session_state.msgs  = []
        st.session_state.met   = []

    if st.button("🗑️ Nueva conversación"):
        st.session_state.msgs = []; st.session_state.met = []; st.rerun()

    if not st.session_state.msgs:
        with st.chat_message("assistant"):
            st.markdown(f"¡Hola! Soy tu **{agent}**. ¿En qué puedo ayudarte?")

    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and "lat" in m:
                c = st.columns(4)
                c[0].caption(f"⏱️ {m['lat']:.2f}s")
                c[1].caption(f"⚡ {m['tps']:.0f} TPS")
                c[2].caption(f"🪙 {m['tok']} tokens")
                if isinstance(m.get("score"), (int, float)):
                    icon = "🟢" if m["score"] >= 8 else "🟡" if m["score"] >= 5 else "🔴"
                    c[3].caption(f"{icon} {m['score']}/10")

    user_in = st.chat_input("Escribe aquí...")
    if user_in:
        if not api_key:
            st.error("Ingresa tu API Key en la barra lateral."); st.stop()

        st.session_state.msgs.append({"role":"user","content":user_in})
        with st.spinner("..."):
            try:
                api_msgs = [{"role":m["role"],"content":m["content"]} for m in st.session_state.msgs]
                content, lat, usage = groq_call(
                    api_key, api_msgs, system=AGENTS[agent],
                    model=model, temperature=temp_a, max_tokens=700
                )
                tps   = usage.total_tokens / lat if lat > 0 else 0
                score = None
                if auto_eval:
                    try:
                        import json
                        eval_p = (f'Evalúa del 1 al 10 esta respuesta de IA.\n'
                                  f'Pregunta: "{user_in}"\nRespuesta: "{content[:400]}"\n'
                                  f'Responde SOLO JSON: {{"score":<n>,"just":"<1 oración>"}}')
                        ev, _, _ = groq_call(api_key, [{"role":"user","content":eval_p}],
                                             model=model, temperature=0.1, max_tokens=80)
                        m_j = re.search(r'\{.*\}', ev, re.DOTALL)
                        if m_j: score = json.loads(m_j.group()).get("score")
                    except: pass

                st.session_state.msgs.append({
                    "role":"assistant","content":content,
                    "lat":lat,"tps":tps,"tok":usage.total_tokens,"score":score
                })
                st.session_state.met.append({"lat":lat,"tps":tps,"tok":usage.total_tokens,"score":score})
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

    if st.session_state.met:
        st.divider()
        st.markdown("### 📊 Métricas de sesión")
        met = st.session_state.met
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱️ Lat. prom.", f"{np.mean([m['lat'] for m in met]):.2f}s")
        c2.metric("⚡ TPS prom.", f"{np.mean([m['tps'] for m in met]):.0f}")
        c3.metric("🪙 Tokens tot.", sum(m['tok'] for m in met))
        scores = [m['score'] for m in met if isinstance(m.get('score'), (int, float))]
        if scores: c4.metric("🎯 Score prom.", f"{np.mean(scores):.1f}/10")
