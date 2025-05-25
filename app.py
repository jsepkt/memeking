from flask import Flask, request, jsonify, send_file, render_template, url_for
import requests
import io
import os
import threading
import sys # Kept for potential future use
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps # For programmatic animation
import imageio # For GIF creation
import math # For animation effects like sine waves
import random # For choosing random animation type
import traceback # For detailed error logging

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
    except (FileNotFoundError, ValueError, IOError):
        count = 0
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
def create_simple_animation_frames(image_bytes, animation_type="wiggle", num_frames=12):
    """
    Creates a list of PIL Image frames for a simple ~1-second animation.
    animation_type: "wiggle", "pulse_scale", "pulse_brightness", "center_jump", 
                    "center_glow", "color_cycle_center", "rotate"
    """
    try:
        base_img_rgba = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        print(f"Error opening base image for animation: {e}")
        traceback.print_exc()
        return []

    frames = []
    width, height = base_img_rgba.size
    center_x, center_y = width // 2, height // 2

    # Central region (used by some effects)
    region_width, region_height = width // 2, height // 2
    region_bbox = (
        max(0, center_x - region_width // 2), max(0, center_y - region_height // 2),
        min(width, center_x + region_width // 2), min(height, center_y + region_height // 2)
    )

    for i in range(num_frames):
        current_frame_canvas_rgba = base_img_rgba.copy() # Start fresh for each frame

        if animation_type == "wiggle":
            max_wiggle_px = int(width * 0.02) 
            offset = int(max_wiggle_px * math.sin(4 * math.pi * i / num_frames))
            current_frame_canvas_rgba = base_img_rgba.transform(
                base_img_rgba.size, Image.AFFINE, (1, 0, offset, 0, 1, 0), resample=Image.BICUBIC
            )

        elif animation_type == "rotate":
            max_angle = 5 
            angle = max_angle * math.sin(2 * math.pi * i / num_frames)
            # Rotate around the center, keeping original size (might crop corners)
            # To avoid cropping, you'd rotate with expand=True then crop/paste onto a new canvas.
            # For this simple version, expand=False is used.
            current_frame_canvas_rgba = base_img_rgba.rotate(angle, resample=Image.BICUBIC, expand=False, center=(center_x, center_y))
            # Ensure the rotated image still has an alpha channel if it's RGBA
            if current_frame_canvas_rgba.mode != 'RGBA' and base_img_rgba.mode == 'RGBA':
                current_frame_canvas_rgba = current_frame_canvas_rgba.convert('RGBA')


        elif animation_type == "pulse_scale":
            scale_factor = 1.0 + 0.05 * math.sin(2 * math.pi * i / num_frames)
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            if new_w <=0 or new_h <=0: new_w, new_h = width, height
            
            scaled_img = base_img_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            temp_canvas = Image.new("RGBA", base_img_rgba.size, (0,0,0,0)) # Transparent
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2
            # Ensure scaled_img has alpha if base had it, for proper pasting
            if scaled_img.mode != 'RGBA' and base_img_rgba.mode == 'RGBA':
                 alpha = base_img_rgba.resize((new_w,new_h), Image.Resampling.LANCZOS).split()[-1]
                 scaled_img = scaled_img.convert("RGB") # Ensure it's RGB before putting alpha
                 scaled_img.putalpha(alpha)

            temp_canvas.paste(scaled_img, (paste_x, paste_y), scaled_img if scaled_img.mode == 'RGBA' else None)
            current_frame_canvas_rgba = temp_canvas

        elif animation_type == "pulse_brightness":
            brightness_factor = 1.0 + 0.15 * math.sin(2 * math.pi * i / num_frames)
            enhancer = ImageEnhance.Brightness(base_img_rgba)
            current_frame_canvas_rgba = enhancer.enhance(brightness_factor)
        
        elif animation_type == "center_jump":
            t_norm = i / float(num_frames -1) if num_frames > 1 else 0
            max_jump_height_px = int(height * 0.08) 
            y_offset = int(max_jump_height_px * (4 * t_norm * (1 - t_norm)))
            
            subject_region = base_img_rgba.crop(region_bbox)
            background_for_jump = base_img_rgba.copy()
            eraser = Image.new('RGBA', (region_bbox[2]-region_bbox[0], region_bbox[3]-region_bbox[1]), (0,0,0,0))
            background_for_jump.paste(eraser, region_bbox, eraser) # Paste transparent region
            
            jumped_pos_y = region_bbox[1] - y_offset
            # Ensure subject_region has alpha for proper pasting if base was RGBA
            if subject_region.mode != 'RGBA' and base_img_rgba.mode == 'RGBA':
                subject_region = subject_region.convert('RGBA') # Should ideally already be if cropped from RGBA

            background_for_jump.paste(subject_region, (region_bbox[0], jumped_pos_y), subject_region if subject_region.mode == 'RGBA' else None)
            current_frame_canvas_rgba = background_for_jump
        
        else: 
            pass 

        final_frame_rgb = Image.new("RGB", current_frame_canvas_rgba.size, (255, 255, 255)) 
        final_frame_rgb.paste(current_frame_canvas_rgba, mask=current_frame_canvas_rgba.split()[3] if current_frame_canvas_rgba.mode == 'RGBA' else None)
        frames.append(final_frame_rgb)
    return frames

# --- Content Filtering Keywords ---
FORBIDDEN_KEYWORDS = [
    "kill", "murder", "slaughter", "decapitate", "behead", "torture", "gore", "guts", "blood", "bloody",
    "massacre", "stab", "shoot", "gun violence", "brutal", "maim", "corpse", "dead body", "funeral",
    "sex", "nude", "naked", "porn", "erotic", "explicit", "xxx", "sexual act", "genitals", "orgasm",
    "masturbation", "intercourse", "rape", "molest", "pedophile", "bestiality", "incest", "lust", "sperm", "vagina", "penis",
    "nazi", "swastika", "kkk", "racist slur", "homophobic slur", "misogynist slur", "heil hitler", 
    "white supremacy", "aryan", "ethnic cleansing", "genocide", "slave",
    "suicide", "self harm", "self-harm", "cutting", "overdose", "depressed", "anorexia", "bulimia", "depicting death",
    "drug making", "bomb making", "meth", "cocaine recipe", "heroin", "lsd", "illegal drugs",
    "child abuse", "child porn", "underage", "animal cruelty", "torturing animals",
    "hate", "fuck", "shit", "bitch", "cunt", "asshole", "motherfucker", 
]

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-meme', methods=['POST'])
def generate_static_meme_route(): 
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
    negative_prompt_terms = "nsfw, nude, naked, sexually explicit, explicit content, violence, gore, blood, weapon, disturbing, graphic, offensive, text, watermark, signature, ugly, deformed, disfigured, poorly drawn hands, poorly drawn face, error, blurry, bad anatomy, extra limbs, missing limbs, text, words, letters, signature, username, artist name, watermark, multiple images, grid, collage"
    
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
    
    cartoon_style_prompt = f"{user_prompt}, funny cartoon style, simple clean vector illustration, vibrant colors, clear outlines, meme character, no text, no signature, centered subject"
    negative_prompt_terms_anim = "realistic, photo, 3d render, complex, detailed background, nsfw, violence, gore, blood, text, watermark, signature, blurry, deformed, grainy, noisy, human, person, people, multiple characters, busy background, dark, shadow, words, letters, multiple images, grid, collage"
    
    payload = {
        "inputs": cartoon_style_prompt,
        "parameters": { "negative_prompt": negative_prompt_terms_anim }
    }
    
    base_image_bytes = query_hf_static_image(payload)

    if not base_image_bytes:
        return jsonify({"error": "Failed to generate base cartoon image for animation. AI model might be busy."}), 503

    try:
        num_animation_frames = 12
        animation_duration_per_frame_ms = 83 

        # For testing, you can fix the animation type.
        # To make it random from a selection:
        available_animations = ["wiggle", "rotate", "pulse_scale", "pulse_brightness", "center_jump"]
        animation_type_to_use = random.choice(available_animations)
        # Or fix it for testing:
        # animation_type_to_use = "wiggle" 
        # animation_type_to_use = "rotate"

        print(f"Applying animation type: {animation_type_to_use}")

        animation_frames = create_simple_animation_frames(
            base_image_bytes, 
            animation_type=animation_type_to_use, 
            num_frames=num_animation_frames
        )
        
        if not animation_frames:
            return jsonify({"error": "Could not create animation frames from the base image."}), 500

        gif_bytes_io = io.BytesIO()
        imageio.mimsave(gif_bytes_io, animation_frames, format='GIF', duration=animation_duration_per_frame_ms, subrectangles=True, palettesize=128, loop=0)
        gif_bytes_out = gif_bytes_io.getvalue()
        
        increment_meme_count()
        return send_file(
            io.BytesIO(gif_bytes_out),
            mimetype='image/gif',
            as_attachment=False,
            download_name=f'memeking_anim_{animation_type_to_use}.gif'
        )
    except Exception as e:
        print(f"Error during static image to GIF animation: {e}")
        traceback.print_exc() 
        if isinstance(e, (IOError, ValueError, TypeError)): 
            return jsonify({"error": "Error processing the image for animation. The base image might be unusual or too complex."}), 500
        return jsonify({"error": "Failed to process the animation into a GIF. Something unexpected happened."}), 500

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
            if COUNT_FILE_DIR != "." and os.path.dirname(COUNT_FILE) and not os.path.exists(os.path.dirname(COUNT_FILE)):
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
        if COUNT_FILE_DIR != ".":
            print(f"MEME COUNT (app.py): Falling back to local directory for count file.")
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
    app.run(host='0.0.0.0', port=port, debug=True)