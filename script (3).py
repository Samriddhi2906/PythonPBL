# ================================
# IMPORT LIBRARIES
# ================================
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("tfidf1.pkl", "rb") as f:
    tfidf1 = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# If you used feature selection
with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)
with open("stopwords.pkl", "rb") as f:
    stopwords = pickle.load(f)

with open("wordnet.pkl", "rb") as f:
    wordnet = pickle.load(f)
    
from preprocessing import preprocess_text
from validation import is_valid_input
# ================================
# USER INPUT
# ================================
user_input = input("\nEnter your review: ")
if not is_valid_input(user_input):
    print("Invalid input")

else:
    clean_input = preprocess_text(user_input)
    # IMPORTANT: same pipeline
    X_input = tfidf.transform([clean_input])
    X_input_selected = selector.transform(X_input)

    prediction = model.predict(X_input_selected)
    result = le.inverse_transform(prediction)

    print("Predicted Sentiment:", result[0])