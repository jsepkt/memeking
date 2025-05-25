from flask import Flask, request, jsonify, send_file, render_template, url_for
import requests
import io
import os
import threading
import sys # Kept for potential future use
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter # For programmatic animation
import imageio # For GIF creation
import math # For animation effects like sine waves

# --- Hugging Face API Configuration ---
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_API_TOKEN:
    print("="*70)
    print("CRITICAL WARNING: HF_API_TOKEN not set in environment variables/secrets.")
    print("Meme generation WILL FAIL.")
    print("Please set this in your hosting environment (e.g., Replit Secrets, Render Environment Variables).")
    print("="*70)

# This will be used for both static images and the base for animated GIFs
STATIC_IMAGE_MODEL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# --- Meme Counter Setup ---
COUNT_FILE_DIR = "."
RENDER_DISK_MOUNT_PATH = "/mnt/data" # For Render persistent disk
if os.path.exists(RENDER_DISK_MOUNT_PATH) and os.access(RENDER_DISK_MOUNT_PATH, os.W_OK):
    COUNT_FILE_DIR = RENDER_DISK_MOUNT_PATH
    print(f"MEME COUNT (app.py): Using persistent disk for count file at {COUNT_FILE_DIR}")
else:
    print(f"MEME COUNT (app.py): Using local directory for count file (will be ephemeral on Render without disk unless path is root).")

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
        except IOError as e: print(f"Error writing count file '{COUNT_FILE}': {e}")
    return count

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='.', static_folder='static')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a-dev-secret-key-change-for-prod-and-set-env-var")


# --- Helper Function for Hugging Face Static Images ---
def query_hf_static_image(payload):
    if not HF_API_TOKEN or not headers:
        print("Cannot query Hugging Face: HF_API_TOKEN is not set or headers not prepared.")
        return None
    # print(f"Sending payload to Hugging Face (Static Image): {payload}") # Uncomment for debug
    response = requests.post(STATIC_IMAGE_MODEL_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        print(f"HF Static Image Error: {response.status_code} - {response.text}")
        return None

# --- Helper for Programmatic Animation ---
def create_simple_animation_frames(image_bytes, animation_type="wiggle", num_frames=12, duration_sec=1.0):
    """
    Creates a list of PIL Image frames for a simple 1-second animation.
    """
    try:
        base_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        print(f"Error opening base image for animation: {e}")
        return []

    frames = []
    width, height = base_img.size

    for i in range(num_frames):
        frame_img_copy = base_img.copy() # Work on a copy for each frame

        # Ensure canvas is transparent for effects that might go outside original bounds
        canvas = Image.new("RGBA", base_img.size, (0, 0, 0, 0))

        if animation_type == "wiggle":
            # Gentle horizontal wiggle, completes two cycles in num_frames
            offset = int(width * 0.02 * math.sin(4 * math.pi * i / num_frames)) # Max 2% of width wiggle
            canvas.paste(frame_img_copy, (offset, 0), frame_img_copy)
        elif animation_type == "pulse_scale":
            scale_factor = 1.0 + 0.05 * math.sin(2 * math.pi * i / num_frames) # Pulse once
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            if new_w <=0 or new_h <=0: # safety check
                new_w, new_h = width, height
            scaled = frame_img_copy.resize((new_w, new_h), Image.Resampling.LANCZOS)
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2
            canvas.paste(scaled, (paste_x, paste_y), scaled)
        elif animation_type == "pulse_brightness":
            brightness_factor = 1.0 + 0.15 * math.sin(2 * math.pi * i / num_frames) # Pulse once, less intense
            enhancer = ImageEnhance.Brightness(frame_img_copy)
            pulsed_img = enhancer.enhance(brightness_factor)
            canvas.paste(pulsed_img, (0,0), pulsed_img) # Paste onto transparent canvas
        else: # Default: no actual animation, just repeat base frame (or could make it a static GIF)
            canvas.paste(frame_img_copy, (0,0), frame_img_copy)
        
        # Convert to RGB for GIF compatibility, handling transparency by pasting on white if needed
        # Or, if GIF supports transparency well with your imageio settings, keep RGBA and handle palette.
        final_frame = Image.new("RGB", canvas.size, (255, 255, 255)) # White background
        final_frame.paste(canvas, mask=canvas) # Paste RGBA using alpha as mask
        frames.append(final_frame)

    return frames


# --- Content Filtering Keywords ---
FORBIDDEN_KEYWORDS = [
    "kill", "murder", "slaughter", "decapitate", "behead", "torture", "gore", "guts", "blood", "bloody",
    "massacre", "stab", "shoot", "gun violence", "brutal", "maim", "corpse", "dead body", "funeral",
    "sex", "nude", "naked", "porn", "erotic", "explicit", "xxx", "sexual act", "genitals", "orgasm",
    "masturbation", "intercourse", "rape", "molest", "pedophile", "bestiality", "incest", "lust",
    "nazi", "swastika", "kkk", "racist slur", "homophobic slur", "misogynist slur", "heil hitler",
    "white supremacy", "aryan", "ethnic cleansing", "genocide",
    "suicide", "self harm", "self-harm", "cutting", "overdose", "depressed", "anorexia", "bulimia",
    "drug making", "bomb making", "meth", "cocaine recipe", "heroin", "lsd",
    "child abuse", "animal cruelty", "torturing animals",
    "hate", "fuck", "shit", "bitch", "cunt", "asshole", # Add common profanities cautiously
    # Add more terms, consider variations, misspellings, leetspeak
]

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# Route for STATIC image memes
@app.route('/generate-meme', methods=['POST'])
def generate_static_meme_route(): # Renamed for clarity, though URL is same
    if not HF_API_TOKEN:
        return jsonify({"error": "Oops! The meme magic is taking a nap. Please try again later."}), 503

    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    user_prompt_lower = user_prompt.lower()

    if not user_prompt:
        return jsonify({"error": "Prompt cannot be empty!"}), 400

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in user_prompt_lower:
            print(f"Blocked inappropriate static prompt: '{user_prompt}' due to keyword: '{keyword}'")
            return jsonify({"error": "Your prompt seems to contain sensitive or inappropriate content. We recommend fun, creative ideas avoiding adult themes, violence, or hate speech."}), 400

    enhanced_prompt = f"{user_prompt}, meme style, funny, crypto coin, digital art, high detail, vibrant colors"
    negative_prompt_terms = "nsfw, nude, naked, sexually explicit, explicit content, violence, gore, blood, weapon, disturbing, graphic, offensive, text, watermark, signature, ugly, deformed, disfigured, poorly drawn hands, poorly drawn face, error, blurry, bad anatomy, extra limbs, missing limbs, text, words, letters, signature, username, artist name, watermark"
    
    payload = {
        "inputs": enhanced_prompt,
        "parameters": { "negative_prompt": negative_prompt_terms }
    }
    image_bytes = query_hf_static_image(payload)

    if image_bytes:
        increment_meme_count()
        return send_file(io.BytesIO(image_bytes), mimetype='image/jpeg')
    else:
        return jsonify({"error": "Failed to generate image. The AI model might be busy or having a moment. Please try again soon!"}), 503

# NEW Route for ANIMATED GIF memes (programmatic animation of a static cartoon image)
@app.route('/generate-animated-meme', methods=['POST'])
def generate_animated_meme_route():
    if not HF_API_TOKEN:
        return jsonify({"error": "Oops! Animated magic is napping. (Admin: Token missing)"}), 503

    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    user_prompt_lower = user_prompt.lower()

    if not user_prompt: return jsonify({"error": "Prompt for animation cannot be empty!"}), 400

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in user_prompt_lower:
            print(f"Blocked inappropriate animated prompt: '{user_prompt}' due to: '{keyword}'")
            return jsonify({"error": "Animated prompt contains sensitive words. Let's keep it fun & friendly with cartoon styles!"}), 400
    
    cartoon_style_prompt = f"{user_prompt}, funny cartoon style, simple clean vector illustration, vibrant colors, clear outlines, meme character, no text, no signature"
    negative_prompt_terms_anim = "realistic, photo, 3d render, complex, detailed background, nsfw, violence, gore, blood, text, watermark, signature, blurry, deformed, grainy, noisy, human, person, people" # More specific negatives for cartoon style
    
    payload = {
        "inputs": cartoon_style_prompt,
        "parameters": { "negative_prompt": negative_prompt_terms_anim }
    }
    
    base_image_bytes = query_hf_static_image(payload) # Generate a static cartoon image first

    if not base_image_bytes:
        return jsonify({"error": "Failed to generate base cartoon image for animation. AI model might be busy."}), 503

    try:
        # --- Create Animated Frames (aim for ~1 second) ---
        num_animation_frames = 12  # e.g., for 12 FPS for 1 second, or 6 FPS if duration is 0.166 per frame
        animation_duration_per_frame_ms = 80 # milliseconds (1000ms / 12 frames = ~83ms. 80ms -> 12.5 FPS)

        # You can let the user choose an animation type in the future via frontend, or pick one randomly
        # For now, let's default to "wiggle" or make it fixed
        animation_type_to_use = "wiggle" # or "pulse_scale" or "pulse_brightness"
        
        animation_frames = create_simple_animation_frames(
            base_image_bytes, 
            animation_type=animation_type_to_use, 
            num_frames=num_animation_frames
        )
        
        if not animation_frames:
            return jsonify({"error": "Could not create animation frames from the base image."}), 500

        # --- Convert Frames to GIF ---
        gif_bytes_io = io.BytesIO()
        imageio.mimsave(gif_bytes_io, animation_frames, format='GIF', duration=animation_duration_per_frame_ms, subrectangles=True, palettesize=128, loop=0)
        gif_bytes_out = gif_bytes_io.getvalue()
        
        increment_meme_count() # Count this as one creation
        return send_file(
            io.BytesIO(gif_bytes_out),
            mimetype='image/gif',
            as_attachment=False, # Display inline
            download_name='animated_cartoon_meme.gif'
        )
    except Exception as e:
        print(f"Error during static image to GIF animation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        if isinstance(e, (IOError, ValueError, TypeError)): 
            return jsonify({"error": "Error processing the image for animation. The base image might be unusual."}), 500
        return jsonify({"error": "Failed to process the animation into a GIF."}), 500

# Other routes remain the same
@app.route('/get-meme-count', methods=['GET'])
def get_meme_count_route():
    with count_lock: count = get_meme_count()
    return jsonify({"count": count})

@app.route('/terms')
def terms_page():
    return render_template('terms.html')

@app.route('/privacy')
def privacy_page():
    return render_template('privacy.html')

if __name__ == '__main__':
    initial_count = 0
    try:
        with count_lock:
            # Ensure directory for COUNT_FILE exists if on a persistent disk subdirectory
            if COUNT_FILE_DIR != "." and not os.path.exists(os.path.dirname(COUNT_FILE)) and os.path.dirname(COUNT_FILE) != '':
                 os.makedirs(os.path.dirname(COUNT_FILE), exist_ok=True)
            with open(COUNT_FILE, "r") as f:
                content = f.read().strip()
                if content: initial_count = int(content)
                else: # File is empty, initialize it
                    with open(COUNT_FILE, "w") as fw: fw.write(str(initial_count))
    except (FileNotFoundError, ValueError): # File not found or invalid content
        with open(COUNT_FILE, "w") as f: f.write(str(initial_count)) # Create/overwrite with 0
    except IOError as e:
        print(f"MEME COUNT (app.py): Error initializing count file '{COUNT_FILE}': {e}")
        print(f"MEME COUNT (app.py): Falling back to local directory for count file (will be ephemeral if not root).")
        # Attempt to use local if persistent disk path failed and COUNT_FILE_DIR was changed
        if COUNT_FILE_DIR != ".":
            COUNT_FILE = "meme_count.txt" 
            try:
                with open(COUNT_FILE, "r") as f_fallback:
                    content = f_fallback.read().strip()
                    if content: initial_count = int(content)
                    else:
                        with open(COUNT_FILE, "w") as fw_fallback: fw_fallback.write(str(initial_count))
            except (FileNotFoundError, ValueError):
                with open(COUNT_FILE, "w") as f_fallback: f_fallback.write(str(initial_count))


    print(f"MEME COUNT (app.py): Initialized. Current count is {initial_count} from '{COUNT_FILE}'")
    
    port = int(os.environ.get("PORT", 8080)) 
    print(f"Starting Flask server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) # Keep debug=True for local/Replit dev