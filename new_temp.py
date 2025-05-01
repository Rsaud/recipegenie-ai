from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import os

# ============ SETUP ============
# 1. Load the recipe dataset
df = pd.read_excel('recs.xlsx')  # Make sure this file exists in the same directory

# 2. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
ingredient_texts = df['المكونات'].tolist()
embeddings = model.encode(ingredient_texts, show_progress_bar=True)

# 3. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. Set Gemini API key
os.environ["API_KEY"] = "YOUR_API_KEY_HERE"  # Replace with your actual Gemini API key
genai.configure(api_key=os.environ["API_KEY"])


# ============ RECIPE GENERATOR ============
def generate_recipe(user_ingredients, top_k=1):
    query_embedding = model.encode([user_ingredients])
    _, indices = index.search(query_embedding, top_k)

    retrieved_recipes = [df.iloc[i]['طريقة التحضير'] for i in indices[0]]
    context = "\n\n".join(retrieved_recipes)

    prompt = f"""
    You are a creative recipe generator and understand many languages. Based on the following ingredients: {user_ingredients},
    and inspired by these example recipes:

    {context}

    Create a new, unique recipe using the provided ingredients. Format it like a cooking blog and in the user input language.
    """

    model_ai = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model_ai.generate_content(prompt)
    return response.text


# ============ FLASK SETUP ============
app = Flask(__name__)

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🍳 Recipe Genie</title>
  <style>
    body { font-family: 'Poppins', sans-serif; background: #f5f7fa; padding: 20px; }
    .chat { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    input, button { padding: 10px; margin-top: 10px; width: 100%; }
    .message { margin-top: 15px; }
    .bot { background: #e6f7f7; padding: 10px; border-radius: 8px; }
    .user { background: #ffe6e6; padding: 10px; border-radius: 8px; text-align: right; }
  </style>
</head>
<body>
  <div class="chat">
    <h2>🍳 Recipe Genie</h2>
    <div id="chat-box"></div>
    <input type="text" id="user-input" placeholder="Ask for a recipe (e.g., 'دجاج و بطاطا')">
    <button onclick="send()">Send</button>
  </div>
<script>
  function send() {
    let msg = document.getElementById('user-input').value;
    if (!msg) return;
    let box = document.getElementById('chat-box');
    box.innerHTML += '<div class="message user">' + msg + '</div>';
    fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'message=' + encodeURIComponent(msg)
    })
    .then(res => res.json())
    .then(data => {
      if (data.recipes) {
        data.recipes.forEach(r => {
          box.innerHTML += '<div class="message bot"><b>' + r.name + '</b><br><i>' + r.country + '</i><br><b>Ingredients:</b> ' + r.ingredients + '<br><b>Instructions:</b><br>' + r.instructions + '</div>';
        });
      } else {
        box.innerHTML += '<div class="message bot">Sorry, no recipes found.</div>';
      }
      document.getElementById('user-input').value = '';
    });
  }
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(html_template)


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided."})

    try:
        result = generate_recipe(user_message, top_k=3)
        return jsonify({
            "recipes": [{
                "name": "Custom Recipe",
                "ingredients": user_message,
                "instructions": result,
                "country": "🌍 AI Kitchen"
            }]
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)