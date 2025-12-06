import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Load model once for performance
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()

# Summarization function
def summarize(text):
    inputs = tokenizer.batch_encode_plus(
        [text],
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=40,
        max_length=150,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ----------------------- STREAMLIT UI -----------------------

st.title("📰 BART Text Summarizer")
st.write("Paste your article or blog below and get a clean summary.")

user_input = st.text_area("Enter Text Here:", height=250)

if st.button("Summarize"):
    if len(user_input.strip()) < 10:
        st.warning("Please enter at least 10 characters.")
    else:
        with st.spinner("Generating summary... ⏳"):
            summary_output = summarize(user_input)

        st.subheader("📌 Summary:")
        st.write(summary_output)
