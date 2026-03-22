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
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.title + ". " + article.text
        except:
            st.error("Failed to extract")
else:
    text = st.text_area("Text:", height=150)

if text and st.button("Analyze"):
    lang = detect(text)
    if lang == "ru":
        st.info("🔄 Translating...")
        ids = translator_tok.encode(text[:1024], return_tensors="pt", truncation=True)
        text = translator_tok.decode(translator.generate(ids)[0], skip_special_tokens=True)
        st.caption(f"Translated: {text[:300]}...")
    
    result = predictor.predict(text)
    explanation = interpreter.explain(text)
    
    st.markdown(f"### Fakeness: {result['confidence'] if result['label']==1 else 1-result['confidence']:.0%}")
    st.progress(int((result['confidence'] if result['label']==1 else 1-result['confidence']) * 100))
    st.text(explanation)
