from flask import Flask, request, jsonify, send_file, render_template, url_for
import requests
import io
import os
import threading # For thread-safe file access
import sys # For exiting if critical env var is missing

# --- Configuration ---
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") # Get token ONLY from Replit Secrets

if not HF_API_TOKEN:
    print("=" * 70)
    print("CRITICAL WARNING: Hugging Face API Token (HF_API_TOKEN) is NOT SET in Replit Secrets.")
    print("Please go to the 'Secrets' tab in your Replit (padlock icon on the left) and add:")
    print("Key: HF_API_TOKEN")
    print("Value: your_actual_hf_token_here (e.g., hf_xxxx...)")
    print("Meme generation will FAIL until this is set.")
    print("=" * 70)
    # You might choose to exit, but Replit might just keep restarting.
    # Letting it run allows you to see the console message.
    # sys.exit("Exiting: HF_API_TOKEN is not set. The application cannot run without it.")


MODEL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
# Headers are prepared only if HF_API_TOKEN is available
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}


# --- Meme Counter Setup ---
# Replit's filesystem is generally persistent for the lifetime of the Repl,
# but major infrastructure changes or if the Repl is forked/copied might affect it.
# For simple V1, a local file is okay. For more robust, Replit Database could be used.
COUNT_FILE_DIR = "." # Store in the current directory within the Repl.
COUNT_FILE = os.path.join(COUNT_FILE_DIR, "meme_count.txt")

count_lock = threading.Lock()

def get_meme_count():
    try:
        with open(COUNT_FILE, "r") as f:
            count = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        count = 0
    return count

def increment_meme_count():
    with count_lock:
        current_val = get_meme_count()
        # print(f"MEME COUNT (app.py): Current value before increment: {current_val}")
        count = current_val + 1
        try:
            with open(COUNT_FILE, "w") as f:
                f.write(str(count))
            # print(f"MEME COUNT (app.py): Incremented and saved. New count: {count}")
        except IOError as e:
            print(f"MEME COUNT (app.py): Error writing to count file '{COUNT_FILE}': {e}")
    return count
# --- End Meme Counter Setup ---

app = Flask(__name__, template_folder='.', static_folder='static')

def query_huggingface_image(payload):
    if not HF_API_TOKEN or not headers:
        print("Cannot query Hugging Face: HF_API_TOKEN is not set or headers not prepared.")
        return None

    # print(f"Sending payload to Hugging Face: {payload}")
    response = requests.post(MODEL_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        # print("Successfully received image from Hugging Face.")
        return response.content
    else:
        error_message = f"Hugging Face API Error: {response.status_code}. "
        try:
            error_details = response.json()
            error_message += str(error_details)
            if response.status_code == 503 and "estimated_time" in error_details:
                error_message += f" Model is likely loading. Estimated time: {error_details.get('estimated_time', 0)}s."
        except requests.exceptions.JSONDecodeError:
            error_message += response.text
        print(error_message) # Log Hugging Face errors
        return None

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-meme', methods=['POST'])
def generate_meme_route():
    if not HF_API_TOKEN:
        print("Server Misconfiguration: HF_API_TOKEN not available for meme generation.")
        return jsonify({"error": "Oops! The meme magic is taking a nap. (Admin: Token missing)"}), 503 # User-friendly error

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is missing!"}), 400

    user_prompt = data['prompt']
    if not user_prompt.strip():
        return jsonify({"error": "Prompt cannot be empty!"}), 400

    enhanced_prompt = f"{user_prompt}, meme style, funny, crypto coin, digital art, high detail, vibrant colors"
    payload = {"inputs": enhanced_prompt}
    image_bytes = query_huggingface_image(payload)

    if image_bytes:
        increment_meme_count()
        return send_file(
            io.BytesIO(image_bytes),
            mimetype='image/jpeg'
        )
    else:
        return jsonify({"error": "Failed to generate image. The AI model might be busy or having a moment. Please try again soon!"}), 503

@app.route('/get-meme-count', methods=['GET'])
def get_meme_count_route():
    with count_lock: 
        count = get_meme_count()
    return jsonify({"count": count})

# --- Run the App on Replit ---
if __name__ == '__main__':
    # Initialize meme_count.txt if it doesn't exist
    initial_count = 0
    try:
        with count_lock:
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

    # Replit sets the PORT environment variable.
    # It's good practice to use '0.0.0.0' to bind to all available interfaces.
    # Replit will typically map an external port to your app's internal port.
    # Using debug=True is common on Replit for easier development feedback.
    # For a more "production-like" Replit, you might set debug=False.
    port = int(os.environ.get('PORT', 8080)) # Replit usually uses 8080 if PORT not explicitly set for Python web server
    print(f"Starting Flask server on host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)