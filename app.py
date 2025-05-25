from flask import Flask, request, jsonify, send_file, render_template, url_for
import requests
import io
import os
import threading
import sys # Kept for potential future use, but not strictly needed now

# --- Hugging Face API Configuration ---
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_API_TOKEN:
    print("="*70)
    print("CRITICAL WARNING: HF_API_TOKEN not set in environment variables/secrets.")
    print("Meme generation WILL FAIL.")
    print("Please set this in your hosting environment (e.g., Replit Secrets, Render Environment Variables).")
    print("="*70)

MODEL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# --- Meme Counter Setup ---
COUNT_FILE_DIR = "."
RENDER_DISK_MOUNT_PATH = "/mnt/data" # For Render persistent disk
if os.path.exists(RENDER_DISK_MOUNT_PATH) and os.access(RENDER_DISK_MOUNT_PATH, os.W_OK):
    COUNT_FILE_DIR = RENDER_DISK_MOUNT_PATH
COUNT_FILE = os.path.join(COUNT_FILE_DIR, "meme_count.txt")
count_lock = threading.Lock()

def get_meme_count():
    try:
        with open(COUNT_FILE, "r") as f: count = int(f.read().strip())
    except: count = 0
    return count

def increment_meme_count():
    with count_lock:
        count = get_meme_count() + 1
        try:
            with open(COUNT_FILE, "w") as f: f.write(str(count))
        except IOError as e: print(f"Error writing count file: {e}")
    return count

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='.', static_folder='static')
# You still need a secret key if you plan to use flash messages or other session-based features,
# but it's less critical without Flask-Login. For now, a placeholder is fine for local.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a-dev-secret-key-change-for-prod")


# --- Helper Function for Hugging Face ---
def query_huggingface_image(payload):
    if not HF_API_TOKEN or not headers:
        print("Cannot query Hugging Face: HF_API_TOKEN is not set or headers not prepared.")
        return None
    response = requests.post(MODEL_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        print(f"HF Error: {response.status_code} - {response.text}")
        return None

# --- Content Filtering Keywords ---
# EXPAND THIS LIST SIGNIFICANTLY!
# Consider categories: violence, sexual, hate, self-harm, illegal activities, etc.
# You can also look for open-source "bad word" lists or profanity filters.
FORBIDDEN_KEYWORDS = [
    # Violence & Gore
    "kill", "murder", "slaughter", "decapitate", "behead", "torture", "gore", "guts", "blood", "bloody",
    "massacre", "stab", "shoot", "gun violence", "brutal", "maim", "corpse", "dead body",
    # Explicit Sexual Content
    "sex", "nude", "naked", "porn", "erotic", "explicit", "xxx", "sexual act", "genitals",
    "orgasm", "masturbation", "intercourse", "rape", "molest", "pedophile", "bestiality",
    # Hate Speech & Discrimination (examples, needs to be much broader)
    "nazi", "swastika", "kkk", "racist slur", "homophobic slur", "misogynist slur", # Replace generic slurs with actual terms
    "white supremacy", "heil hitler",
    # Self-Harm
    "suicide", "self harm", "cutting", "overdose",
    # Illegal Activities (examples)
    "drug making", "bomb making", "meth", "cocaine recipe",
    # Other Potentially Problematic
    "child abuse", "animal cruelty",
    # Add more... this is just a starting point.
    # Consider misspellings, leetspeak, and common variations.
]

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-meme', methods=['POST'])
def generate_meme_route():
    if not HF_API_TOKEN:
        return jsonify({"error": "Oops! The meme magic is taking a nap. Please try again later."}), 503

    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    user_prompt_lower = user_prompt.lower() # For case-insensitive matching

    if not user_prompt:
        return jsonify({"error": "Prompt cannot be empty!"}), 400

    # --- Backend Keyword Filtering ---
    for keyword in FORBIDDEN_KEYWORDS:
        # Use word boundaries for more precise matching if desired (e.g., re.search(r'\b' + keyword + r'\b', ...))
        # For simplicity, direct 'in' check first.
        if keyword in user_prompt_lower:
            print(f"Blocked inappropriate prompt: '{user_prompt}' due to keyword: '{keyword}'") # Log attempt
            return jsonify({"error": "Your prompt seems to contain sensitive or inappropriate content. We recommend avoiding topics related to violence, adult themes, or hate speech. Please try a more creative and fun idea!"}), 400
    # --- End Backend Keyword Filtering ---

    enhanced_prompt = f"{user_prompt}, meme style, funny, crypto coin, digital art, high detail, vibrant colors"
    # Negative prompts are still a good idea
    negative_prompt_terms = "nsfw, nude, naked, sexually explicit, explicit content, violence, gore, blood, weapon, disturbing, graphic, offensive, text, watermark, signature, ugly, deformed, disfigured, poorly drawn hands, poorly drawn face, error, blurry, bad anatomy, extra limbs, missing limbs"
    
    payload = {
        "inputs": enhanced_prompt,
        "parameters": { "negative_prompt": negative_prompt_terms }
    }
    image_bytes = query_huggingface_image(payload)

    if image_bytes:
        increment_meme_count()
        return send_file(io.BytesIO(image_bytes), mimetype='image/jpeg')
    else:
        return jsonify({"error": "Failed to generate image. The AI model might be busy or having a moment. Please try again soon!"}), 503

@app.route('/get-meme-count', methods=['GET'])
def get_meme_count_route():
    with count_lock: count = get_meme_count()
    return jsonify({"count": count})

@app.route('/terms')
def terms_page():
    return render_template('terms.html') # You'll need to create/update terms.html

@app.route('/privacy')
def privacy_page():
    return render_template('privacy.html') # You'll need to create/update privacy.html


if __name__ == '__main__':
    initial_count = 0
    try:
        with count_lock:
            if COUNT_FILE_DIR != "." and not os.path.exists(os.path.dirname(COUNT_FILE)):
                 os.makedirs(os.path.dirname(COUNT_FILE), exist_ok=True)
            with open(COUNT_FILE, "r") as f:
                content = f.read().strip()
                if content: initial_count = int(content)
                else:
                    with open(COUNT_FILE, "w") as fw: fw.write(str(initial_count))
    except (FileNotFoundError, ValueError):
        with open(COUNT_FILE, "w") as f: f.write(str(initial_count))
    except IOError as e:
        print(f"MEME COUNT (app.py): Error initializing count file '{COUNT_FILE}': {e}")

    print(f"MEME COUNT (app.py): Initialized. Current count is {initial_count} from '{COUNT_FILE}'")
    
    port = int(os.environ.get("PORT", 8080)) # Replit/Cloud Run often use 8080 or set PORT
    print(f"Starting Flask server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)