import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.header(" UMAP Topic Clusters")
st.markdown("Visualized topic clusters from interview data using TF-IDF + UMAP.")

# Load data & models
df = pd.read_csv("data/train.csv")
#df = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/train.csv")
df['Labels'] = df['Labels'].astype(str)
reducer = joblib.load("umap_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Recompute embeddings
X = vectorizer.transform(df["Interview Text"])
X_embedded = reducer.transform(X)
df['x'] = X_embedded[:, 0]
df['y'] = X_embedded[:, 1]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='Labels', palette='tab10', ax=ax)
plt.title("UMAP Projection of Interview Topics")
st.pyplot(fig)