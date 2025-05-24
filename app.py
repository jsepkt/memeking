from flask import Flask, request, jsonify, send_file, render_template, url_for
import requests
import io
import os
import threading # For thread-safe file access
import sys # For exiting if critical env var is missing

# --- Configuration ---
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") # Get token ONLY from environment variable

if not HF_API_TOKEN:
    # This will be visible in Render's logs if the env var isn't set
    print("CRITICAL ERROR: Hugging Face API Token (HF_API_TOKEN) environment variable not found.")
    # Optionally, you can make the app exit if the token isn't set,
    # as it's critical for functionality.
    # sys.exit("Exiting: HF_API_TOKEN is not set. The application cannot run.")
    # For now, we'll let it proceed, but API calls will fail.
    # On Render, ensure HF_API_TOKEN is set in the environment variables for the service.

MODEL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
# Ensure headers are only prepared if HF_API_TOKEN is available
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}


# --- Meme Counter Setup ---
# Check if running on Render and using a persistent disk for the counter
RENDER_DISK_MOUNT_PATH = "/mnt/data" # Standard Render disk mount path
if os.path.exists(RENDER_DISK_MOUNT_PATH) and os.access(RENDER_DISK_MOUNT_PATH, os.W_OK):
    COUNT_FILE_DIR = RENDER_DISK_MOUNT_PATH
    print(f"MEME COUNT (app.py): Using persistent disk for count file at {COUNT_FILE_DIR}")
else:
    COUNT_FILE_DIR = "." # Current directory (ephemeral for Render web service)
    print(f"MEME COUNT (app.py): Using local directory for count file (will be ephemeral on Render without disk).")

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
        print(f"MEME COUNT (app.py): Current value before increment: {current_val}") # DEBUG
        count = current_val + 1
        try:
            with open(COUNT_FILE, "w") as f:
                f.write(str(count))
            print(f"MEME COUNT (app.py): Incremented and saved. New count: {count}") # DEBUG
        except IOError as e:
            print(f"MEME COUNT (app.py): Error writing to count file '{COUNT_FILE}': {e}") # DEBUG
    return count
# --- End Meme Counter Setup ---

app = Flask(__name__, template_folder='.', static_folder='static')

def query_huggingface_image(payload):
    if not HF_API_TOKEN: # Check if token is missing before making API call
        print("Cannot query Hugging Face: HF_API_TOKEN is not set.")
        return None

    print(f"Sending payload to Hugging Face: {payload}") # DEBUG
    response = requests.post(MODEL_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        print("Successfully received image from Hugging Face.") # DEBUG
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
        print(error_message) # DEBUG
        return None

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-meme', methods=['POST'])
def generate_meme_route():
    if not HF_API_TOKEN: # Prevent generation if token isn't configured
        return jsonify({"error": "Server configuration error: API token missing."}), 500

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
        print("MEME COUNT (app.py): Attempting to increment count...") # DEBUG
        increment_meme_count()
        return send_file(
            io.BytesIO(image_bytes),
            mimetype='image/jpeg'
        )
    else:
        return jsonify({"error": "Failed to generate image. The AI model might be loading or an API error occurred. Please try again in a minute."}), 503

@app.route('/get-meme-count', methods=['GET'])
def get_meme_count_route():
    with count_lock: 
        count = get_meme_count()
    print(f"MEME COUNT (app.py): Serving count: {count}") # DEBUG
    return jsonify({"count": count})

# --- Run the App ---
if __name__ == '__main__':
    if not HF_API_TOKEN:
        print("*" * 60)
        print("WARNING: HF_API_TOKEN environment variable is not set.")
        print("The application will run, but meme generation will FAIL.")
        print("For local development, set this variable in your shell or using a .env file.")
        print("For deployment (e.g., on Render), set it in the service's environment variables.")
        print("*" * 60)

    initial_count = 0
    try:
        with count_lock:
            if COUNT_FILE_DIR != ".":
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
        print(f"MEME COUNT (app.py): Falling back to local directory for count file (will be ephemeral).")
        COUNT_FILE = "meme_count.txt"
        with count_lock:
            try:
                with open(COUNT_FILE, "r") as f_fallback:
                    content = f_fallback.read().strip()
                    if content: initial_count = int(content)
                    else:
                        with open(COUNT_FILE, "w") as fw_fallback: fw_fallback.write(str(initial_count))
            except (FileNotFoundError, ValueError):
                with open(COUNT_FILE, "w") as f_fallback: f_fallback.write(str(initial_count))

    print(f"MEME COUNT (app.py): Initialized/checked. Current count is {initial_count} from '{COUNT_FILE}'")

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)