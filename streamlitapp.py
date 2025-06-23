import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ✅ Set page config before any other Streamlit commands
st.set_page_config(page_title="News Article Classifier", page_icon="📰")

# Load model/tokenizer once
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("./saved_fake_news_model")
    tokenizer = BertTokenizer.from_pretrained("./saved_fake_news_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("📰 News Article Classification")
st.markdown("Enter a news statement or headline to check if it's **real** or **fake**.")

text = st.text_area("📝 News Statement:", height=150)

if st.button("🔍 Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=1).item()

        label_map = {0: "❌ Fake News", 1: "✅ Real News"}
        st.success(f"**Prediction:** {label_map[prediction]}")
