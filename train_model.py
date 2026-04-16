import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("dataset/spam.csv", encoding='latin-1')

# 🔍 Check columns (for safety)
print("Columns in dataset:", data.columns)

# ✅ Handle common dataset format
if 'v1' in data.columns and 'v2' in data.columns:
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
elif 'label' in data.columns and 'text' in data.columns:
    data = data[['label', 'text']]
else:
    raise Exception("Dataset format not recognized. Check column names.")

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Remove any missing values
data.dropna(inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")
from sklearn.metrics import accuracy_score

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))