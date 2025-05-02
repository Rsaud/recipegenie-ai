from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv
import concurrent.futures


load_dotenv()

app = Flask(__name__)
CORS(app, origins=["https://recipe-geni.com", "https://recipegenie-ai-2.onrender.com"])

# Configure Gemini API
genai.configure(api_key=os.environ.get("API_KEY"))


# Load dataset
df = pd.read_excel('final_food_rec_with_type.xlsx')
documents = []
for _, row in df.iterrows():
    doc = f"""البلد: {row['البلد']}
النوع: {row['النوع']}
الاسم: {row['الاسم']}
المكونات: {row['المكونات']}
الطريقة: {row['الطريقة']}"""
    documents.append(doc)

# Prepare embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# Greetings
greetings = {
    "ar": ["السلام عليكم", "مرحبا", "اهلا", "اهلا وسهلا"],
    "en": ["hi", "hello", "welcome"]
}

# Food keywords
food_keywords = [
    "لحم", "دجاج", "سمك", "خضار", "جبن", "رز", "معكرونة", "مكرونة", "خبز", "بطاطس", "تفاح", "برتقال", "طماطم",
    "كوسا", "باذنجان", "عسل", "ملح", "سكر", "بهارات", "لبن", "زبادي", "شوربة", "فطيرة", "كعك", "حليب",
    "شوكولاتة", "فانيليا", "ماء", "زيت", "خل", "تمر", "موز", "فراولة", "عنب", "كريمة", "فلفل", "بصل", "ثوم",
    "mint", "cheese", "bread", "meat", "chicken", "fish", "rice", "pasta", "soup", "salad", "apple", "orange",
    "tomato", "onion", "garlic", "honey", "sugar", "salt", "butter", "cake", "cream", "milk", "yogurt"
]

# Detect greeting
def detect_greeting(user_message):
    user_message = user_message.strip().lower()
    for lang, words in greetings.items():
        for word in words:
            if word in user_message:
                return lang
    return None

# Detect language
def detect_language(user_message):
    arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
    if any(c in arabic_chars for c in user_message):
        return "ar"
    else:
        return "en"

# Check if input related to food
def is_food_related(user_message):
    user_message = user_message.lower()
    for food_word in food_keywords:
        if food_word in user_message:
            return True
    return False
    

# Recipe generation
def generate_recipe(user_ingredients, user_lang="ar", top_k=3):
    model_ai = genai.GenerativeModel('gemini-1.5-flash-latest')
    query_embedding = model.encode([user_ingredients])
    D, indices = index.search(query_embedding, top_k)
    retrieved_recipes = [documents[i] for i in indices[0]]
    context = "\n\n".join(retrieved_recipes)

    if user_lang == "ar":
        instruction_language = "اكتب الوصفة باللغة العربية وبأسلوب مدونة طبخ أنيق."
    else:
        instruction_language = "Write the recipe in English in a stylish cooking blog format."

    prompt = f"""
    You are a creative recipe generator. Based on the following ingredients: {user_ingredients},
    and inspired by these example recipes:

    {context}

    {instruction_language}
    """

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(model_ai.generate_content, prompt)
            response = future.result(timeout=20)  # Timeout in 20 seconds
            return response.text.strip()
    except concurrent.futures.TimeoutError:
        return "❌ Gemini API timed out. Try again."
    except Exception as e:
        print("🔴 Gemini call failed:", str(e))
        return "❌ Unexpected error occurred."


    
    # response = model_ai.generate_content(prompt)
    # return response.text.strip()


# Routes
@app.route('/')
def home():
    return render_template('index.html')
@app.route("/ai-generator")
def ai_generator():
    return render_template("ai-generator.html")


@app.route("/chat", methods=["POST"])
def chat():
    print("🔵 Chat endpoint called")
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided."})

    user_message_lower = user_message.strip().lower()

    # Detect greeting
    greeting_lang = detect_greeting(user_message)
    if greeting_lang:
        if greeting_lang == "ar":
            if "السلام عليكم" in user_message_lower:
                reply = "وعليكم السلام! كيف يمكنني مساعدتك؟"
            elif "مرحبا" in user_message_lower or "اهلا" in user_message_lower or "اهلا وسهلا" in user_message_lower:
                reply = "أهلا وسهلا! تفضل، كيف يمكنني مساعدتك؟"
            else:
                reply = "أهلا بك! كيف يمكنني خدمتك؟"
        elif greeting_lang == "en":
            if "hi" in user_message_lower or "hello" in user_message_lower:
                reply = "Hello! How can I assist you?"
            elif "welcome" in user_message_lower:
                reply = "Welcome! How can I help you?"
            else:
                reply = "Hi there! How can I assist you?"

        return jsonify({
            "recipes": [{
                "name": "Greeting",
                "country": "🤖",
                "ingredients": user_message,
                "instructions": reply
            }]
        })

    # Detect language
    detected_lang = detect_language(user_message)

    # Check if food-related
    if not is_food_related(user_message):
        simple_error = "يرجى كتابة مكون أو طعام صحيح." if detected_lang == "ar" else "Please enter a valid food ingredient."
        return jsonify({
            "recipes": [{
                "name": "Invalid Input",
                "country": "🤖",
                "ingredients": user_message,
                "instructions": simple_error
            }]
        })

    # Try generating recipe
    try:
        generated_recipe = generate_recipe(user_message, user_lang=detected_lang, top_k=3)

        if not generated_recipe.strip():
            raise ValueError("Empty recipe generated.")

        return jsonify({
            "recipes": [{
                "name": "Custom Recipe",
                "country": "🌍 AI Kitchen",
                "ingredients": user_message,
                "instructions": generated_recipe
            }]
        })
    except Exception:
        fallback_error = "عذرًا، حدث خطأ. حاول مرة أخرى!" if detected_lang == "ar" else "Sorry, an error occurred. Please try again!"
        return jsonify({
            "recipes": [{
                "name": "Error",
                "country": "🤖",
                "ingredients": user_message,
                "instructions": fallback_error
            }]
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

