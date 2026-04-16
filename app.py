from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import re
import requests
import os
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Supabase setup (from .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
print("URL:", SUPABASE_URL)  # optional debug
print("KEY:", SUPABASE_KEY)  # optional debug

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# In-memory history (can later move to DB)
history = []

# Suspicious keywords
suspicious_words_list = ['win', 'money', 'prize', 'click', 'urgent', 'offer', 'free']


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text


# AI API (OpenRouter)
def check_with_ai(message):
    api_key = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Scam Detector"
    }

    data = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {
                "role": "user",
                "content": f"Classify this message as scam or safe and explain briefly: {message}"
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return "AI response error"

    except Exception as e:
        print("AI ERROR:", e)
        return "AI analysis unavailable"


# Home (protected)
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", history=history)


# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    message = request.form["message"]

    # URL detection
    url_pattern = r'(https?://\S+|www\.\S+|\S+\.com|\S+\.in)'
    has_url = re.search(url_pattern, message)

    # Clean message
    clean_msg = clean_text(message)

    # ML Prediction
    message_vec = vectorizer.transform([clean_msg])
    prediction = model.predict(message_vec)[0]
    probability = model.predict_proba(message_vec)[0]

    spam_prob = probability[1] * 100

    if prediction == 1:
        ml_result = "Scam Message"
        confidence = spam_prob
    else:
        ml_result = "Safe Message"
        confidence = 100 - spam_prob

    # Suspicious words
    words = clean_msg.split()
    found_words = [w for w in words if w in suspicious_words_list]

    # AI Analysis
    ai_result = check_with_ai(message)
    ai_flag = False

    if ai_result:
        ai_lower = ai_result.lower()
        if ("scam" in ai_lower or 
            "fraud" in ai_lower or 
            "phishing" in ai_lower or 
            "suspicious" in ai_lower):
            ai_flag = True

    # Final Decision (Hybrid)
    if ai_flag:
        final_result = "Scam Message (AI Verified)"
        confidence = max(confidence, 85)

    elif has_url and found_words:
        final_result = "Scam Message (Rule-Based Detection)"
        confidence = max(confidence, 80)

    else:
        final_result = ml_result

    # URL warning
    url_warning = "Contains suspicious link" if has_url else None

    # Save history
    history.append((message, final_result))

    return render_template(
        "index.html",
        prediction=final_result,
        confidence=round(confidence, 2),
        words=found_words,
        url_warning=url_warning,
        history=history,
        user_message=message,
        ai_result=ai_result
    )


# Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            if response.user:
                return redirect(url_for("login"))
            else:
                return "Signup failed"

        except Exception as e:
            print("SIGNUP ERROR:", e)
            return "Signup failed: " + str(e)

    return render_template("signup.html")


# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response.user:
                session["user"] = email
                return redirect(url_for("home"))
            else:
                return "Invalid email or password"

        except Exception as e:
            print("LOGIN ERROR:", e)
            return "Login failed: " + str(e)

    return render_template("login.html")


# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# Run app
if __name__ == "__main__":
    app.run(debug=True)