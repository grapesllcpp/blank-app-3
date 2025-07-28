import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

st.title("Instagram Caption Classifier")

# --- Upload CSV File ------------------------------------------------------
uploaded_file = st.file_uploader("Upload your Instagram CSV file", type="csv")

# --- Default Keyword Categories -------------------------------------------
st.sidebar.header("Keyword Categories")
default_categories = {
    'fashion': ['dress', 'style', 'lycra', 'look', 'outfit', 'crepe'],
    'weather': ['weather', 'sunny', 'warm', 'cold', 'rain'],
    'selfpromo': ['custom', 'order', 'shop', 'dm', 'link in bio'],
}

categories = {}
for cat, words in default_categories.items():
    keywords = st.sidebar.text_input(f"Keywords for {cat}", ", ".join(words))
    categories[cat] = [w.strip() for w in keywords.split(',') if w.strip()]

if uploaded_file:
    raw = pd.read_csv(uploaded_file)
    raw = raw.rename(columns={'shortcode': 'ID', 'caption': 'Context'})

    # --- Sentence Tokenization --------------------------------------------
    records = []
    for _, r in raw.iterrows():
        if pd.notnull(r['Context']):
            for i, sent in enumerate(sent_tokenize(r['Context']), start=1):
                records.append({'ID': r['ID'], 'Context': r['Context'], 'Sentence ID': i, 'Statement': sent})

    df = pd.DataFrame(records)

    # --- Classification ---------------------------------------------------
    def classify(sentence: str, rules: dict) -> str:
        s = sentence.lower()
        hits = [cat for cat, kws in rules.items() if any(k.lower() in s for k in kws)]
        return ';'.join(hits) if hits else 'other'

    df['Category'] = df['Statement'].apply(lambda x: classify(x, categories))

    st.subheader("Classified Sentences")
    st.dataframe(df.head(10))
       # --- Download ---------------------------------------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Classified Data", data=csv, file_name='ig_posts_classified.csv', mime='text/csv')
