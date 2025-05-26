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
from werkzeug.utils import secure_filename # For handling uploaded filenames safely

# --- Hugging Face API Configuration ---
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_API_TOKEN:
    print("="*70)
    print("CRITICAL WARNING: HF_API_TOKEN not set in environment variables/secrets.")
    print("Meme generation WILL FAIL.")
    print("Please set this in your hosting environment (e.g., Replit Secrets, Render Environment Variables).")
    print("="*70)

STATIC_IMAGE_MODEL_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# --- Meme Counter Setup ---
COUNT_FILE_DIR = "."
RENDER_DISK_MOUNT_PATH = "/mnt/data" 
if os.path.exists(RENDER_DISK_MOUNT_PATH) and os.access(RENDER_DISK_MOUNT_PATH, os.W_OK):
    COUNT_FILE_DIR = RENDER_DISK_MOUNT_PATH
    print(f"MEME COUNT (app.py): Using persistent disk for count file at {COUNT_FILE_DIR}")
else:
    print(f"MEME COUNT (app.py): Using local directory for count file.")

COUNT_FILE = os.path.join(COUNT_FILE_DIR, "meme_count.txt")
count_lock = threading.Lock()

def get_meme_count():
    try:
        with open(COUNT_FILE, "r") as f: count = int(f.read().strip())
    except (FileNotFoundError, ValueError, IOError): count = 0
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

# --- Configuration for User Image Uploads ---
# UPLOAD_FOLDER_NAME = 'user_uploads' # If you were saving files to disk
# app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, UPLOAD_FOLDER_NAME)
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])
    
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # GIF input will use its first frame
MAX_UPLOAD_SIZE_MB = 5 
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Helper Function for Hugging Face Static Images ---
def query_hf_static_image(payload):
    if not HF_API_TOKEN or not headers:
        print("Cannot query Hugging Face: HF_API_TOKEN is not set or headers not prepared.")
        return None
    response = requests.post(STATIC_IMAGE_MODEL_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        print(f"HF Static Image Error: {response.status_code} - {response.text}")
        return None

# --- Helper for Programmatic Animation (MODIFIED TO RETURN RGBA FRAMES) ---
def create_simple_animation_frames(image_bytes, animation_type="wiggle", num_frames=12):
    """
    Creates a list of PIL RGBA Image frames for a simple animation.
    animation_type: "wiggle", "pulse_scale", "pulse_brightness", "center_jump", 
                    "center_glow", "color_cycle_center", "rotate"
    """
    try:
        # Open the image and ensure it's RGBA. If it's a GIF, .open() gets the first frame.
        base_img_rgba = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        print(f"Error opening base image for animation: {e}")
        traceback.print_exc()
        return []

    frames = []
    width, height = base_img_rgba.size
    center_x, center_y = width // 2, height // 2
    region_width, region_height = width // 2, height // 2
    region_bbox = (
        max(0, center_x - region_width // 2), max(0, center_y - region_height // 2),
        min(width, center_x + region_width // 2), min(height, center_y + region_height // 2)
    )

    for i in range(num_frames):
        current_frame_canvas_rgba = base_img_rgba.copy()

        if animation_type == "wiggle":
            max_wiggle_px = int(width * 0.02) 
            offset = int(max_wiggle_px * math.sin(4 * math.pi * i / num_frames))
            current_frame_canvas_rgba = base_img_rgba.transform(
                base_img_rgba.size, Image.AFFINE, (1, 0, offset, 0, 1, 0), resample=Image.BICUBIC
            )
        elif animation_type == "rotate":
            max_angle = 5 
            angle = max_angle * math.sin(2 * math.pi * i / num_frames)
            current_frame_canvas_rgba = base_img_rgba.rotate(angle, resample=Image.BICUBIC, expand=False, center=(center_x, center_y))
            if current_frame_canvas_rgba.mode != 'RGBA': # Ensure it stays RGBA after rotate
                current_frame_canvas_rgba = current_frame_canvas_rgba.convert('RGBA')
        elif animation_type == "pulse_scale":
            scale_factor = 1.0 + 0.05 * math.sin(2 * math.pi * i / num_frames)
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            if new_w <=0 or new_h <=0: new_w, new_h = width, height
            scaled_img = base_img_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            temp_canvas = Image.new("RGBA", base_img_rgba.size, (0,0,0,0)) 
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2
            temp_canvas.paste(scaled_img, (paste_x, paste_y), scaled_img) # scaled_img is already RGBA
            current_frame_canvas_rgba = temp_canvas
        elif animation_type == "pulse_brightness":
            brightness_factor = 1.0 + 0.15 * math.sin(2 * math.pi * i / num_frames)
            # Brightness works on RGB part, preserve alpha
            rgb_part = base_img_rgba.convert("RGB")
            alpha_part = base_img_rgba.split()[-1]
            enhancer = ImageEnhance.Brightness(rgb_part)
            enhanced_rgb = enhancer.enhance(brightness_factor)
            current_frame_canvas_rgba = Image.merge("RGBA", (*enhanced_rgb.split(), alpha_part))
        elif animation_type == "center_jump":
            t_norm = i / float(num_frames -1) if num_frames > 1 else 0
            max_jump_height_px = int(height * 0.08) 
            y_offset = int(max_jump_height_px * (4 * t_norm * (1 - t_norm)))
            subject_region = base_img_rgba.crop(region_bbox)
            background_for_jump = base_img_rgba.copy()
            eraser = Image.new('RGBA', (region_bbox[2]-region_bbox[0], region_bbox[3]-region_bbox[1]), (0,0,0,0))
            background_for_jump.paste(eraser, region_bbox, eraser) 
            jumped_pos_y = region_bbox[1] - y_offset
            background_for_jump.paste(subject_region, (region_bbox[0], jumped_pos_y), subject_region)
            current_frame_canvas_rgba = background_for_jump
        else: 
            pass # Original image if no animation type matches

        # IMPORTANT CHANGE: Append the RGBA frame directly
        frames.append(current_frame_canvas_rgba.copy()) # .copy() is good practice in loops
    return frames

# --- New Helper Functions for User Image Transformation ---
def create_cartoon_effect(pil_image):
    """Applies a basic cartoon effect to a PIL Image, returns RGBA."""
    img_input_rgba = pil_image.convert("RGBA") # Ensure input is RGBA
    img_rgb = img_input_rgba.convert("RGB") # Effects work better on RGB

    # Reduce colors (posterize)
    img_posterized = ImageOps.posterize(img_rgb, 4) # 4 bits per channel

    # Detect edges
    img_edges = img_posterized.filter(ImageFilter.FIND_EDGES)
    img_edges = img_edges.convert("L") # Grayscale
    img_edges = ImageOps.invert(img_edges) # Invert: edges are black
    
    # Make edges thicker for cartoon look
    img_edges = img_edges.filter(ImageFilter.MaxFilter(3)) 
    
    # Convert posterized back to RGBA and apply original alpha
    img_cartoon_rgba = img_posterized.convert("RGBA")
    img_cartoon_rgba.putalpha(img_input_rgba.split()[-1]) # Preserve original alpha

    # Overlay black lines: create a black image, use edges as alpha mask
    black_lines_overlay = Image.new("RGBA", img_cartoon_rgba.size, (0, 0, 0, 0)) # Transparent black
    # Use inverted edges (white lines on black) as mask for solid black color
    # Or, draw black pixels where edges are black
    edge_mask_for_black = ImageOps.invert(img_edges) # Lines are now white

    # Create a solid black image of the same size
    black_color_img = Image.new("RGB", img_cartoon_rgba.size, "black")
    
    # Composite the black lines onto the cartoon image
    # Where edge_mask_for_black is white, use black_color_img, else use img_cartoon_rgba
    final_cartoon = Image.composite(black_color_img, img_cartoon_rgba, edge_mask_for_black)
    
    return final_cartoon.convert("RGBA") # Ensure final is RGBA

def apply_zoom_center_effect(pil_image, zoom_factor=1.3, output_size=None):
    """Zooms into the center of an image. Returns RGBA."""
    img_rgba = pil_image.convert("RGBA")
    width, height = img_rgba.size
    
    if output_size is None:
        output_size = (width, height)

    crop_width = int(width / zoom_factor)
    crop_height = int(height / zoom_factor)
    if crop_width < 1 or crop_height < 1: # Avoid zero or negative crop dimensions
        return img_rgba # Return original if zoom makes it too small

    crop_x = (width - crop_width) // 2
    crop_y = (height - crop_height) // 2
    
    cropped_img = img_rgba.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    zoomed_img = cropped_img.resize(output_size, Image.Resampling.LANCZOS)
    return zoomed_img

def crop_to_circle(pil_image):
    """Crops a PIL image to a circle with a transparent background. Returns RGBA."""
    img_rgba = pil_image.convert("RGBA")
    width, height = img_rgba.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    img_rgba.putalpha(mask)
    return img_rgba

def process_uploaded_image_to_emoji_impl(image_bytes, effect_type="cartoon", target_size=(128,128)):
    """Processes uploaded image bytes into an emoji (PNG)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Error opening uploaded image for emoji: {e}")
        traceback.print_exc()
        return None

    processed_img = img.copy().convert("RGBA") # Start with RGBA version

    if effect_type == "cartoon":
        processed_img = create_cartoon_effect(processed_img)
    elif effect_type == "zoom_center": # Simple "bulge" like effect
        processed_img = apply_zoom_center_effect(processed_img, zoom_factor=1.3)
    # Add more effects here as elif blocks
    
    processed_img = crop_to_circle(processed_img)
    processed_img = processed_img.resize(target_size, Image.Resampling.LANCZOS)

    byte_arr = io.BytesIO()
    processed_img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

def process_uploaded_image_to_gif_impl(image_bytes, animation_type="wiggle", num_frames=12, duration_ms=83):
    """Processes uploaded image bytes into an animated GIF."""
    # create_simple_animation_frames now returns RGBA PIL frames
    animation_frames_rgba = create_simple_animation_frames(
        image_bytes, 
        animation_type=animation_type, 
        num_frames=num_frames
    )

    if not animation_frames_rgba:
        print("Failed to create RGBA animation frames for uploaded image GIF.")
        return None

    gif_bytes_io = io.BytesIO()
    try:
        # imageio with Pillow backend handles RGBA frames for GIF, managing palette transparency
        imageio.mimsave(
            gif_bytes_io, 
            animation_frames_rgba, 
            format='GIF', 
            duration=duration_ms, 
            subrectangles=True, 
            palettesize=128, 
            loop=0,
            # Disposal mode 2 is often good for transparent GIFs if backgrounds change
            # Pillow default might be fine, but can specify: disposal=2 
        )
        return gif_bytes_io.getvalue()
    except Exception as e:
        print(f"Error saving GIF from uploaded image: {e}")
        traceback.print_exc()
        return None

# --- Content Filtering Keywords (Keep as is) ---
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
    return render_template('index.html') # You'll need to update index.html for the new feature

@app.route('/generate-meme', methods=['POST'])
def generate_static_meme_route(): 
    if not HF_API_TOKEN:
        return jsonify({"error": "Oops! The meme magic is taking a nap. Please try again later."}), 503
    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    if not user_prompt: return jsonify({"error": "Prompt cannot be empty!"}), 400
    user_prompt_lower = user_prompt.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in user_prompt_lower:
            print(f"Blocked inappropriate static prompt: '{user_prompt}' due to keyword: '{keyword}'")
            return jsonify({"error": "Your prompt seems to contain sensitive content."}), 400
    enhanced_prompt = f"{user_prompt}, meme style, funny, crypto coin, digital art, high detail, vibrant colors"
    negative_prompt_terms = "nsfw, nude, naked, sexually explicit, explicit content, violence, gore, blood, weapon, disturbing, graphic, offensive, text, watermark, signature, ugly, deformed, disfigured, poorly drawn hands, poorly drawn face, error, blurry, bad anatomy, extra limbs, missing limbs, text, words, letters, signature, username, artist name, watermark, multiple images, grid, collage"
    payload = {"inputs": enhanced_prompt, "parameters": { "negative_prompt": negative_prompt_terms }}
    image_bytes = query_hf_static_image(payload)
    if image_bytes:
        increment_meme_count()
        return send_file(io.BytesIO(image_bytes), mimetype='image/jpeg')
    else:
        return jsonify({"error": "Failed to generate image. AI busy. Try again!"}), 503

@app.route('/generate-animated-meme', methods=['POST'])
def generate_animated_meme_route():
    if not HF_API_TOKEN:
        return jsonify({"error": "Oops! Animated magic is napping. (Admin: Token missing)"}), 503
    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    if not user_prompt: return jsonify({"error": "Prompt for animation cannot be empty!"}), 400
    user_prompt_lower = user_prompt.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in user_prompt_lower:
            print(f"Blocked inappropriate animated prompt: '{user_prompt}' due to: '{keyword}'")
            return jsonify({"error": "Animated prompt contains sensitive words. Keep it fun!"}), 400
    cartoon_style_prompt = f"{user_prompt}, funny cartoon style, simple clean vector illustration, vibrant colors, clear outlines, meme character, no text, no signature, centered subject"
    negative_prompt_terms_anim = "realistic, photo, 3d render, complex, detailed background, nsfw, violence, gore, blood, text, watermark, signature, blurry, deformed, grainy, noisy, human, person, people, multiple characters, busy background, dark, shadow, words, letters, multiple images, grid, collage"
    payload = {"inputs": cartoon_style_prompt, "parameters": { "negative_prompt": negative_prompt_terms_anim }}
    base_image_bytes = query_hf_static_image(payload)
    if not base_image_bytes:
        return jsonify({"error": "Failed to generate base cartoon image for animation. AI model might be busy."}), 503
    try:
        num_animation_frames = 12
        animation_duration_per_frame_ms = 83 
        available_animations = ["wiggle", "rotate", "pulse_scale", "pulse_brightness", "center_jump"]
        animation_type_to_use = random.choice(available_animations)
        print(f"Applying animation type: {animation_type_to_use}")
        animation_frames = create_simple_animation_frames(base_image_bytes, animation_type=animation_type_to_use, num_frames=num_animation_frames)
        if not animation_frames: return jsonify({"error": "Could not create animation frames."}), 500
        gif_bytes_io = io.BytesIO()
        imageio.mimsave(gif_bytes_io, animation_frames, format='GIF', duration=animation_duration_per_frame_ms, subrectangles=True, palettesize=128, loop=0)
        gif_bytes_out = gif_bytes_io.getvalue()
        increment_meme_count()
        return send_file(io.BytesIO(gif_bytes_out), mimetype='image/gif', as_attachment=False, download_name=f'memeking_anim_{animation_type_to_use}.gif')
    except Exception as e:
        print(f"Error during static image to GIF animation: {e}")
        traceback.print_exc() 
        return jsonify({"error": "Failed to process animation into GIF."}), 500

# --- NEW ROUTE FOR USER IMAGE UPLOAD AND TRANSFORMATION ---
@app.route('/upload-and-transform', methods=['POST'])
def upload_and_transform_route():
    if 'image_file' not in request.files:
        return jsonify({"error": "No image file part in the request"}), 400
    
    file = request.files['image_file']
    # transform_type from form: 'emoji' or 'gif'
    transform_type = request.form.get('transform_type', 'emoji').lower()
    # effect from form: e.g., 'cartoon', 'zoom_center' for emoji; 'wiggle', 'pulse_scale' for gif
    effect = request.form.get('effect', '').lower()

    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read() # Read file into memory

            # Check size after reading, to prevent oversized files from crashing Pillow
            if len(image_bytes) > app.config['MAX_CONTENT_LENGTH']:
                 return jsonify({"error": f"File too large. Max size: {MAX_UPLOAD_SIZE_MB}MB"}), 413

            processed_image_bytes = None
            mimetype = ''
            download_name = 'transformed_image'

            if transform_type == 'emoji':
                # Default effect for emoji if not specified or invalid
                valid_emoji_effects = ["cartoon", "zoom_center"]
                if effect not in valid_emoji_effects:
                    effect = "cartoon" 
                
                processed_image_bytes = process_uploaded_image_to_emoji_impl(image_bytes, effect_type=effect)
                mimetype = 'image/png'
                download_name = f'custom_emoji_{effect}.png'
            
            elif transform_type == 'gif':
                # Default animation for gif if not specified or invalid
                available_gif_animations = ["wiggle", "rotate", "pulse_scale", "pulse_brightness", "center_jump"]
                if effect not in available_gif_animations:
                    effect = random.choice(available_gif_animations)

                processed_image_bytes = process_uploaded_image_to_gif_impl(image_bytes, animation_type=effect)
                mimetype = 'image/gif'
                download_name = f'custom_anim_{effect}.gif'
            
            else:
                return jsonify({"error": "Invalid transform_type. Must be 'emoji' or 'gif'."}), 400

            if processed_image_bytes:
                # You might want to increment a different counter or no counter for this feature
                # increment_meme_count() # Or a new counter: increment_transformed_image_count()
                print(f"Successfully transformed uploaded image: type='{transform_type}', effect='{effect}'")
                return send_file(
                    io.BytesIO(processed_image_bytes),
                    mimetype=mimetype,
                    as_attachment=False, 
                    download_name=download_name
                )
            else:
                return jsonify({"error": f"Failed to process image for {transform_type} with effect {effect}."}), 500

        except Exception as e:
            print(f"Error processing uploaded image at /upload-and-transform: {e}")
            traceback.print_exc()
            # Check for common Pillow errors if possible, e.g., UnidentifiedImageError
            if "cannot identify image file" in str(e).lower():
                 return jsonify({"error": "Could not open or read image file. It might be corrupted or an unsupported format."}), 400
            return jsonify({"error": "An internal error occurred while processing the image."}), 500
    else:
        allowed_ext_str = ", ".join(ALLOWED_EXTENSIONS)
        return jsonify({"error": f"File type not allowed. Please upload {allowed_ext_str}."}), 400


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
    app.run(host='0.0.0.0', port=port, debug=True) # Set debug=False for production