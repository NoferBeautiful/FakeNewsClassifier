from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from langdetect import detect
from newspaper import Article
from PIL import Image
import streamlit as st
import requests
import torch

st.markdown("## Prediction of Fakeness by Given URL")
background = Image.open('logo.jpg')
st.image(background)

st.markdown(f"### Article URL")
text = st.text_area("Insert some url here", 
        value="https://en.globes.co.il/en/article-yandex-looks-to-expand-activities-in-israel-1001406519")

@st.cache(allow_output_mutation=True)
def get_models_and_tokenizers():
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.load_state_dict(torch.load('./model.pth', map_location='cpu'))

    model_name_translator = "facebook/wmt19-ru-en"
    tokenizer_translator = FSMTTokenizer.from_pretrained(model_name_translator)
    model_translator = FSMTForConditionalGeneration.from_pretrained(model_name_translator)
    model_translator.eval()
    return model, tokenizer, model_translator, tokenizer_translator

model, tokenizer, model_translator, tokenizer_translator = get_models_and_tokenizers()

article = Article(text)
article.download()
article.parse()
concated_text = article.title + '. ' + article.text
lang = detect(concated_text)

st.markdown(f"### Language detection")

if lang == 'ru':
    st.markdown(f"The language of this article is {lang.upper()} so we translated it!")
    with st.spinner('Waiting for translation'):
        input_ids = tokenizer_translator.encode(concated_text, 
            return_tensors="pt", max_length=512, truncation=True)
        outputs = model_translator.generate(input_ids)
        decoded = tokenizer_translator.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### Translated Text")
        st.markdown(f"{decoded[:777]}")
        concated_text = decoded
else:
    st.markdown(f"The language of this article for sure:  {lang.upper()}!")

    st.markdown("### Extracted Text")
    st.markdown(f"{concated_text[:777]}")

tokens_info = tokenizer(concated_text, truncation=True, return_tensors="pt")
with torch.no_grad():
    raw_predictions = model(**tokens_info)
softmaxed = int(torch.nn.functional.softmax(raw_predictions.logits[0], dim=0)[1] * 100)
st.markdown("### Fakeness Prediction")
st.progress(softmaxed)
st.markdown(f"This is fake by **{softmaxed}%**!")
if (softmaxed > 70):
    st.error('We would not trust this text!')
elif (softmaxed > 40):
    st.warning('We are not sure about this text!')
else:
    st.success('We would trust this text!')