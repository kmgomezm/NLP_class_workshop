# 🧠 Taller NLP & LLMs — EAFIT

**Maestría en Ciencia de Datos | Universidad EAFIT — Periodo 2026-1**  
Docente: Jorge Iván Padilla-Buriticá

Aplicación interactiva en Streamlit para explorar la evolución del NLP: desde representaciones clásicas hasta agentes conversacionales con Llama 3.3.

---

## 🚀 Demo en Vivo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](TU_URL_STREAMLIT_AQUI)

---

## 📋 Contenido de la App

| Sección | Descripción |
|---------|-------------|
| 🏠 Inicio | Presentación del taller y guía de configuración |
| 📝 Quiz Conceptual | 4 preguntas con retroalimentación detallada |
| 🔤 Tokenización & Encoding | Word-level, BPE, char-level, One-Hot visual |
| 📊 Vectorización Clásica | Matrices BoW y TF-IDF interactivas con corpus personalizable |
| 🔗 Modelado de Secuencias | N-grams, comparativa RNN/LSTM/GRU/Transformer, HMM vs CRF |
| 🤖 Laboratorio LLM | Playground de temperatura/top-p, visualización Softmax |
| 💬 Agente Conversacional | Agente especializado con Llama 3.3, métricas de rendimiento |

---

## ⚙️ Instalación Local

```bash
git clone https://github.com/TU_USUARIO/nlp-taller-eafit.git
cd nlp-taller-eafit
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Configuración de API Key

Para las funcionalidades de LLM necesitas una **Groq API Key** gratuita:

1. Regístrate en [console.groq.com](https://console.groq.com)
2. Crea un proyecto y genera tu API Key
3. En la app, ingrésala en la barra lateral (campo de tipo `password`)

> ⚠️ **Seguridad:** La clave nunca se almacena — solo vive en la sesión de Streamlit.

### Opción alternativa: `secrets.toml` (para deploy)

Crea el archivo `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_tu_clave_aqui"
```

---

## 🗂️ Estructura del Proyecto

```
nlp-taller-eafit/
├── app.py                    # Entry point principal
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml           # Tema de Streamlit
└── pages_modules/
    ├── __init__.py
    ├── home.py               # Página de inicio
    ├── quiz.py               # Quiz conceptual
    ├── tokenization.py       # Tokenización & Encoding
    ├── vectorization.py      # BoW & TF-IDF
    ├── sequences.py          # N-grams & Arquitecturas
    ├── llm_lab.py            # Playground de temperatura
    └── agent.py              # Agente conversacional
```

---

## 🧪 Tecnologías

- **Frontend:** Streamlit
- **LLM API:** Groq (Llama 3.3 70B / Llama 3 8B)
- **NLP:** scikit-learn, NLTK, pandas
- **Visualización:** Plotly
- **Lenguaje:** Python 3.10+

---

## 📚 Referencias

- [Groq API Documentation](https://console.groq.com/docs)
- [Llama 3.3 Model Card](https://ai.meta.com/blog/meta-llama-3/)
- [Streamlit Documentation](https://docs.streamlit.io)
- Jurafsky & Martin — *Speech and Language Processing* (3rd ed.)

---

*"El lenguaje es el vestido del pensamiento." — Samuel Johnson*
