# # Section D: Visualization of Topic Clusters

# # Imports

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# import umap.umap_ as umap  # Correct import for UMAP
# import joblib  # For saving models

# # 1. Loading dataset
# df = pd.read_csv("/data/train.csv")
# #df = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/train.csv")

# # Ensuring labels are treated as strings for categorical coloring in plots
# df['Labels'] = df['Labels'].astype(str)

# # 2. TF-IDF Vectorization

# # Convert raw text into TF-IDF features using unigrams and bigrams
# vectorizer = TfidfVectorizer(
#     max_features=5000,  # Limit vocabulary size
#     ngram_range=(1, 2), # Use unigrams and bigrams
#     stop_words="english" # Remove English stopwords
# )

# # Applying vectorization to the Interview Text

# X = vectorizer.fit_transform(df["Interview Text"])

# # 3. UMAP dimensionality reduction

# # Reduce high-dimensional TF-IDF vectors to 2D space for visualization
# reducer = umap.UMAP(random_state=42)
# X_embedded = reducer.fit_transform(X)

# # 4. Add UMAP coordinates to DataFrame

# # Store 2D UMAP coordinates as new columns in the original dataframe
# df['x'] = X_embedded[:, 0]
# df['y'] = X_embedded[:, 1]

# # 5. Plot

# # Create a scatterplot of the UMAP-reduced data colored by label

# plt.figure(figsize=(10, 7))
# sns.scatterplot(data=df, x='x', y='y', hue='Labels', palette='tab10')

# # Title and formatting

# plt.title("UMAP Topic Clustering of Interview Transcripts")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

# # 6. Save plot to local as a png file
# plt.savefig("umap_topic_clustering.png", dpi=300)
# plt.close()

# # 7. Save model as .pkl for reuse 
# joblib.dump(reducer, "umap_model.pkl")
# joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# print("Done! Saved plot, UMAP model, and vectorizer.")

# Section D: Visualization of Topic Clusters

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap  # Correct import for UMAP
import joblib  # For saving models

# 1. Loading dataset
df = pd.read_csv("data/train.csv")  # Adjusted to use relative path for cloud deployment

# Ensuring labels are treated as strings for categorical coloring in plots
df['Labels'] = df['Labels'].astype(str)

# 2. TF-IDF Vectorization
# Convert raw text into TF-IDF features using unigrams and bigrams
vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit vocabulary size
    ngram_range=(1, 2), # Use unigrams and bigrams
    stop_words="english" # Remove English stopwords
)

# Applying vectorization to the Interview Text
X = vectorizer.fit_transform(df["Interview Text"])

# 3. UMAP dimensionality reduction
# Reduce high-dimensional TF-IDF vectors to 2D space for visualization
reducer = umap.UMAP(random_state=42)
X_embedded = reducer.fit_transform(X)

# 4. Add UMAP coordinates to DataFrame
# Store 2D UMAP coordinates as new columns in the original dataframe
df['x'] = X_embedded[:, 0]
df['y'] = X_embedded[:, 1]

# 5. Plot
# Create a scatterplot of the UMAP-reduced data colored by label
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='x', y='y', hue='Labels', palette='tab10')

# Title and formatting
plt.title("UMAP Topic Clustering of Interview Transcripts")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 6. Save plot to local as a png file
plt.savefig("umap_topic_clustering.png", dpi=300)
plt.close()

# 7. Save model and vectorizer as .pkl for reuse
joblib.dump(reducer, "umap_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Done! Saved plot, UMAP model, and vectorizer.")
