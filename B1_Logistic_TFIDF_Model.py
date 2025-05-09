# Section B: Model 1 - logistic regression with TF-IDF and subword-level analyzer

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Sci-kit imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2

# Handling class imbalance
from imblearn.over_sampling import RandomOverSampler

# 1. Load and preprocess

# Loading data
#df = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/train.csv")
df = pd.read_csv("data/train.csv")
# Ensuring labels are integers
df['Labels'] = df['Labels'].astype(int)

# Split dataset into train and test sets --> 90% train, 10% test
X_train, X_test, y_train, y_test = train_test_split(
    df["Interview Text"], df["Labels"],
    stratify=df["Labels"], test_size=0.1, random_state=42
)

# 2. TF-IDF with subword-level analyzer and more features

# Vectorize text using character n-grams (subword analysis) for robustness against typos and rare words
vectorizer = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2), #uni and bi-grams
    analyzer='char_wb',  # character n-grams analysis
    sublinear_tf=True, # apply sublinear term frequency scaling
    min_df=1 # include all terms appearing at least once
)

# Transform train and test data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Handling class Imbalance by Oversampling

# Applying random oversampling to balance the class distribution
ros = RandomOverSampler(random_state=42)
X_train_vec, y_train = ros.fit_resample(X_train_vec, y_train)
print(f"✅ After oversampling: {Counter(y_train)}")

# 4. Feature selection (Helps with overfitting)

# Select top 10,000 features based on chi-squared test
selector = SelectKBest(chi2, k=10000)
X_train_vec = selector.fit_transform(X_train_vec, y_train)
X_test_vec = selector.transform(X_test_vec)

# 5. Logistic Regression with cross - validation

# LogisticRegressionCV performs model selection and cross-validation
lr_model = LogisticRegressionCV(
    Cs=10, #range of inverse regularization strengths
    cv=10, # 10 fold cross validation
    class_weight='balanced', # adjust weights inversely proportional to class frequencies
    max_iter=3000, # allow more iterations for convergence
    scoring='f1_weighted', # optimize weighted F1 score
    solver='lbfgs', # efficient solver for multinomial loss
    multi_class='multinomial', # handle multi-class problems natively
    random_state=42
)

# Train model

lr_model.fit(X_train_vec, y_train)

# Make Predictions
lr_preds = lr_model.predict(X_test_vec)

# 6. Evaluation of Model Performance

# Calculating Accuracy and F1 score
acc = accuracy_score(y_test, lr_preds)
f1 = f1_score(y_test, lr_preds, average='weighted')
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ Weighted F1 Score: {f1:.4f}\n")
print(classification_report(y_test, lr_preds, zero_division=0))

# 7. Confusion Matrix Visualization
cm = confusion_matrix(y_test, lr_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Final TF-IDF Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 8. Feature Importance     

# Retrieve feature names that survived feature selection
feature_names = vectorizer.get_feature_names_out()
selected_features = selector.get_support(indices=True)
filtered_features = feature_names[selected_features]

# Print top 10 influential features per class
for i, class_coef in enumerate(lr_model.coef_):
    top_features = sorted(zip(class_coef, filtered_features), reverse=True)[:10]
    print(f"\n Top features for class {i}:")
    print(", ".join([word for _, word in top_features]))
