import streamlit as st
from newspaper import Article
from langdetect import detect
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from model.predictor import BertPredictor
from model.interpretability import AttentionInterpreter

st.set_page_config(page_title="Fake News Classifier", page_icon="📰")
st.title("📰 Fake News Classifier")

MODELS = ["baseline_frozen_encoder", "baseline_frozen_encoder_trump", "baseline_unfrozen_encoder"]

@st.cache_resource
def load_translator():
    tok = FSMTTokenizer.from_pretrained("facebook/wmt19-ru-en")
    model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-ru-en")
    return tok, model

@st.cache_resource
def load_predictor(name):
    return BertPredictor(name), AttentionInterpreter(name)

model_name = st.selectbox("Model:", MODELS)
predictor, interpreter = load_predictor(model_name)
translator_tok, translator = load_translator()

input_type = st.radio("Input:", ["URL", "Text"], horizontal=True)
text = None

if input_type == "URL":
    url = st.text_input("URL:")
    if url:
        with st.spinner("Extracting article..."):
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.title + ". " + article.text
                st.success("Article extracted!")
            except:
                st.error("Failed to extract article")
else:
    text = st.text_area("Text:", height=150)

if text:
    lang = detect(text)
    if lang == "ru":
        with st.spinner("🔄 Translating from Russian..."):
            ids = translator_tok.encode(text[:1024], return_tensors="pt", truncation=True)
            text = translator_tok.decode(translator.generate(ids)[0], skip_special_tokens=True)
        st.info(f"**Translated text:** {text}")
    
    with st.spinner("Analyzing..."):
        result = predictor.predict(text)
        data = interpreter.explain_detailed(text)
    
    fake_prob = data['confidence'] if data['label'] == 1 else 1 - data['confidence']
    st.markdown(f"### Fakeness: {fake_prob:.0%}")
    st.progress(int(fake_prob * 100))
    
    st.markdown("---")
    st.markdown("### 📊 Interpretation")
    st.markdown(f"**Punctuation attention:** {data['punct']:.4f}")
    st.markdown(f"**Stop words attention:** {data['stop']:.4f}")
    
    st.markdown("**Top important words:**")
    top_words = ", ".join([f"`{w}` ({s:.3f})" for w, s in data['top_words'][:10]])
    st.markdown(top_words)
    
    st.markdown("---")
    st.markdown("### 🔍 Text with attention highlighting")
    
    scores = [s for _, s in data['all_tokens']]
    max_score = max(scores) if scores else 1
    
    html_parts = []
    for token, score in data['all_tokens']:
        intensity = min(score / max_score, 1.0)
        r = int(255 * intensity)
        g = int(255 * (1 - intensity * 0.5))
        b = int(100 * (1 - intensity))
        color = f"rgb({r},{g},{b})"
        display = token[2:] if token.startswith("##") else " " + token
        html_parts.append(f'<span style="background-color:{color};padding:2px;border-radius:3px">{display}</span>')
    
    st.markdown("".join(html_parts), unsafe_allow_html=True)
