from flask import Flask, request, jsonify, send_file, render_template, url_for
import requests
import io
import os
import threading # For thread-safe file access

# --- Configuration ---
# It's better practice to set HF_API_TOKEN as an environment variable.
# For local testing, you can hardcode it, but be careful with version control.
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "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") # Your token
MODEL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Meme Counter Setup ---
# Check if running on Render and using a persistent disk for the counter
RENDER_DISK_MOUNT_PATH = "/mnt/data" # Standard Render disk mount path
if os.path.exists(RENDER_DISK_MOUNT_PATH) and os.access(RENDER_DISK_MOUNT_PATH, os.W_OK):
    COUNT_FILE_DIR = RENDER_DISK_MOUNT_PATH
    # Optional: Create a subdirectory on the disk if desired
    # COUNT_FILE_DIR = os.path.join(RENDER_DISK_MOUNT_PATH, "memeking_data")
    # if not os.path.exists(COUNT_FILE_DIR):
    #     os.makedirs(COUNT_FILE_DIR, exist_ok=True)
    print(f"MEME COUNT (app.py): Using persistent disk for count file at {COUNT_FILE_DIR}")
else:
    COUNT_FILE_DIR = "." # Current directory (ephemeral for Render web service)
    print(f"MEME COUNT (app.py): Using local directory for count file (will be ephemeral on Render without disk).")

COUNT_FILE = os.path.join(COUNT_FILE_DIR, "meme_count.txt")

# Use a lock for thread-safe access to the count file
count_lock = threading.Lock()

def get_meme_count():
    # This function is called within a lock, so no need for separate lock here
    try:
        with open(COUNT_FILE, "r") as f:
            count = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        count = 0 # Default to 0 if file doesn't exist or content is invalid
    return count

def increment_meme_count():
    with count_lock:
        current_val = get_meme_count() # Get current count
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

# Initialize the Flask app
# Explicitly defining template_folder and static_folder
app = Flask(__name__, template_folder='.', static_folder='static')


def query_huggingface_image(payload):
    """Sends a prompt to Hugging Face and gets image bytes back."""
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
    """Serves the main HTML page."""
    # If you were using the environment variable method for PUMP_FUN_LINK:
    # pump_fun_link = os.environ.get("PUMP_FUN_LINK", "DEFAULT_PUMP_FUN_LINK_IF_NOT_SET")
    # return render_template('index.html', pump_fun_link=pump_fun_link)
    return render_template('index.html')

@app.route('/generate-meme', methods=['POST'])
def generate_meme_route():
    """Handles the meme generation request from the frontend."""
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
        increment_meme_count() # Increment count on successful generation
        return send_file(
            io.BytesIO(image_bytes),
            mimetype='image/jpeg'
        )
    else:
        return jsonify({"error": "Failed to generate image. The AI model might be loading or an API error occurred. Please try again in a minute."}), 503

# New route to get the meme count
@app.route('/get-meme-count', methods=['GET'])
def get_meme_count_route():
    with count_lock: 
        count = get_meme_count()
    print(f"MEME COUNT (app.py): Serving count: {count}") # DEBUG
    return jsonify({"count": count})


# --- Run the App ---
if __name__ == '__main__':
    # Ensure the count file is writable by initializing it if it doesn't exist or is empty
    initial_count = 0
    try:
        with count_lock:
            # Try to create the directory for COUNT_FILE if it's on a disk and doesn't exist
            # This is more relevant if COUNT_FILE_DIR is a subdirectory on the disk
            if COUNT_FILE_DIR != ".": # i.e., if we are trying to use a specific path like /mnt/data
                os.makedirs(os.path.dirname(COUNT_FILE), exist_ok=True)

            with open(COUNT_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    initial_count = int(content)
                else:
                    with open(COUNT_FILE, "w") as fw:
                        fw.write(str(initial_count))
    except (FileNotFoundError, ValueError):
        with open(COUNT_FILE, "w") as f:
            f.write(str(initial_count))
    except IOError as e: # Catch potential errors creating directory or file on disk
        print(f"MEME COUNT (app.py): Error initializing count file '{COUNT_FILE}': {e}")
        print(f"MEME COUNT (app.py): Falling back to local directory for count file (will be ephemeral).")
        COUNT_FILE = "meme_count.txt" # Fallback to local if disk path fails
        # Retry initialization with local fallback
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

    # Get port from environment variable (for Render, Heroku, etc.) or default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    # For production, Gunicorn (or another WSGI server) will run the app.
    # app.run() is mainly for local development.
    # When deploying, the `startCommand` in render.yaml (e.g., "gunicorn app:app") takes precedence.
    # Setting debug=False is crucial for any production-like environment.
    # host='0.0.0.0' makes the server accessible externally (required by most hosting platforms).
    print(f"Starting Flask app on host 0.0.0.0, port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False for production readiness