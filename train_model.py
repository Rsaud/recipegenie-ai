# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load and clean your dataset
df = pd.read_csv('dataset.csv')

# Fill NA values and convert to string
text_columns = ['اسم الوصفة', 'الدولة', 'المكونات الأساسية', 'التصنيف', 'طريقة التحضير']
for col in text_columns:
    df[col] = df[col].fillna('').astype(str)

# Create combined features
df['combined_features'] = df['اسم الوصفة'] + ' ' + df['المكونات الأساسية'] + ' ' + df['التصنيف']

# Initialize and fit TF-IDF Vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Save the model files
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

# Save the dataframe for reference
df.to_pickle('recipes_dataframe.pkl')

print("Model training complete! Saved:")
print("- tfidf_vectorizer.pkl")
print("- tfidf_matrix.pkl") 
print("- recipes_dataframe.pkl")