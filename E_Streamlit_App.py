import streamlit as st
import pandas as pd
import torch
import joblib
import plotly.express as px
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---- Page setup ----
st.set_page_config(page_title="Interview NLP App", layout="wide")
st.title("🎤 Interview Transcript Classifier & Generator")

# ---- Load models and data ----
df = pd.read_csv("/data/train.csv").dropna()
#df = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/train.csv").dropna()
df["Labels"] = df["Labels"].astype(str)

# Load UMAP and vectorizer
umap_model = joblib.load("DistilBERT_Model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
X_vec = tfidf_vectorizer.transform(df["Interview Text"])
X_embedded = umap_model.transform(X_vec)
df["x"], df["y"] = X_embedded[:, 0], X_embedded[:, 1]

# Load classification model and tokenizer
classifier = DistilBertForSequenceClassification.from_pretrained("results")
tokenizer = DistilBertTokenizerFast.from_pretrained("results")

# ---- Classify transcript function ----
def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = classifier(**inputs)
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    return pred_label

# ---- Label mapping ----
label_map = {
    0: "pre_game_expectations",
    1: "post_game_reaction",
    2: "in_game_analysis",
    3: "career_reflection",
    4: "controversial_opinion",
    5: "injury_report",
    6: "training_insight",
    7: "off_field"
}
inv_label_map = {v: k for k, v in label_map.items()}

# ---- Load GPT-2 text generator ----
if "generator" not in st.session_state:
    st.session_state.generator = pipeline("text-generation", model="gpt2")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📊 Topic Clusters", "🧠 Transcript Classifier", "💬 Q&A Generator"])

# === Tab 1: Topic Clusters ===
with tab1:
    st.subheader("📌 UMAP Topic Cluster Visualization")
    fig = px.scatter(
        df, x="x", y="y", color="Labels",
        hover_data=["Interview Text"], title="UMAP Projection of Interview Topics"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Number of Topic Categories:**")
    st.success(f"{df['Labels'].nunique()} unique categories detected.")

    st.markdown("### 📝 Approach Description")
    st.write("""
    We vectorized interview transcripts using TF-IDF and applied UMAP for dimensionality reduction. 
    This allowed visualizing the semantic distribution of categories and discovering topic clusters interactively.
    """)

# === Tab 2: Transcript Classification ===
with tab2:
    st.subheader("🧠 Predict Interview Transcript Category")
    user_input = st.text_area("Enter the full transcript here:")
    if st.button("Classify Transcript"):
        if user_input.strip():
            pred = predict_category(user_input)
            label = label_map[pred]
            st.success(f"Predicted Category: **{label}**")
        else:
            st.warning("Please enter valid text.")

# === Tab 3: Text Generation ===
with tab3:
    st.subheader("💬 Interview Q&A Generator")
    category = st.selectbox("Choose Interview Category", list(inv_label_map.keys()))
    question = st.text_input("Enter a question:")

    if st.button("Generate Response"):
        prompt = f"Category: {category}\nQuestion: {question}\nAnswer:"
        output = st.session_state.generator(prompt, max_length=80, num_return_sequences=1)[0]["generated_text"]
        answer = output.split("Answer:")[-1].strip()
        st.text_area("Generated Answer", value=answer, height=120)

    st.markdown("### 🧪 Sample Q&A Examples:")
    samples = [
        ("pre_game_expectations", "What are your expectations today?"),
        ("post_game_reaction", "What are your thoughts on the match?"),
        ("career_reflection", "What's been a defining moment in your career?"),
        ("in_game_analysis", "What changed at halftime?"),
        ("controversial_opinion", "Did you agree with the red card?")
    ]
    for cat, q in samples:
        prompt = f"Category: {cat}\nQuestion: {q}\nAnswer:"
        response = st.session_state.generator(prompt, max_length=80, num_return_sequences=1)[0]['generated_text']
        st.markdown(f"**Q ({cat})**: {q}")
        st.markdown(f"> **A**: {response.split('Answer:')[-1].strip()}")

    st.markdown("---")
    st.markdown("### 🧠 Ethical Reflection")
    st.write("""
    AI-generated content can help sports journalists quickly generate interview drafts and mock content.
    However, misuse can lead to misattributed or misleading quotes. 
    Ethical guidelines must ensure AI responses are properly disclosed and not mistaken for real speech.
    """)
