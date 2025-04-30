from langdetect import detect
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # This is the missing import
from langdetect import detect
from flask import Flask, request, jsonify, render_template

# Load data with explicit handling of missing values
df = pd.read_csv('dataset.csv', na_filter=False)

def clean_text(text):
    # Convert to string if not already
    text = str(text) if not isinstance(text, str) else text
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s,]', '', text)
    text = re.sub(r'\d+\.', '', text)
    return text.strip().lower()  # Now safe to call lower()

# Apply cleaning to all text columns
text_columns = ['اسم الوصفة', 'الدولة', 'المكونات الأساسية', 'التصنيف', 'طريقة التحضير']
for col in text_columns:
    df[col] = df[col].apply(clean_text)

# Create combined features safely
df['combined_features'] = (
    df['اسم الوصفة'] + ' ' + 
    df['المكونات الأساسية'] + ' ' + 
    df['التصنيف']
)

class RecipeChatbot:
    def __init__(self):
        try:
            # Load the trained model components
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf = pickle.load(f)
            
            with open('tfidf_matrix.pkl', 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            
            self.df = pd.read_pickle('recipes_dataframe.pkl')
            
            # Ensure all text columns are strings
            text_cols = ['اسم الوصفة', 'الدولة', 'المكونات الأساسية', 'التصنيف', 'طريقة التحضير']
            for col in text_cols:
                self.df[col] = self.df[col].astype(str).fillna('')
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def get_response(self, user_input):
        try:
            # Ensure input is string
            user_input = str(user_input).strip()
            if not user_input:
                return {'error': 'Empty input'}
            
            # Detect language safely
            try:
                lang = detect(user_input)
                lang = lang if lang in ['en', 'ar'] else 'en'
            except:
                lang = 'en'
            
            # Vectorize input and calculate similarity
            user_vec = self.tfidf.transform([user_input.lower()])
            similarity_scores = cosine_similarity(user_vec, self.tfidf_matrix)
            
            # Get top 3 matches
            top_indices = np.argsort(similarity_scores[0])[-3:][::-1]
            
            # Prepare response
            response = {
                'language': lang,
                'recipes': []
            }
            
            for idx in top_indices:
                recipe = {
                    'name': self.df.iloc[idx]['اسم الوصفة'],
                    'country': self.df.iloc[idx]['الدولة'],
                    'category': self.df.iloc[idx]['التصنيف'],
                    'ingredients': self.df.iloc[idx]['المكونات الأساسية'],
                    'instructions': self.df.iloc[idx]['طريقة التحضير']
                }
                response['recipes'].append(recipe)
            
            return response
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return {
                'language': 'en',
                'error': 'Sorry, I encountered an error processing your request'
            }