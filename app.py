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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
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

# --- Helper for Programmatic Animation (Returns RGBA FRAMES) ---
def create_simple_animation_frames(image_bytes, animation_type="wiggle", num_frames=12):
    try:
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
            if current_frame_canvas_rgba.mode != 'RGBA':
                current_frame_canvas_rgba = current_frame_canvas_rgba.convert('RGBA')
        elif animation_type == "pulse_scale":
            scale_factor = 1.0 + 0.05 * math.sin(2 * math.pi * i / num_frames)
            new_w, new_h = int(width * scale_factor), int(height * scale_factor)
            if new_w <=0 or new_h <=0: new_w, new_h = width, height
            scaled_img = base_img_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            temp_canvas = Image.new("RGBA", base_img_rgba.size, (0,0,0,0))
            paste_x, paste_y = (width - new_w) // 2, (height - new_h) // 2
            temp_canvas.paste(scaled_img, (paste_x, paste_y), scaled_img)
            current_frame_canvas_rgba = temp_canvas
        elif animation_type == "pulse_brightness":
            brightness_factor = 1.0 + 0.15 * math.sin(2 * math.pi * i / num_frames)
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
            # CORRECTED VARIABLE NAME HERE:
            jumped_pos_y = region_bbox[1] - y_offset
            background_for_jump.paste(subject_region, (region_bbox[0], jumped_pos_y), subject_region)
            current_frame_canvas_rgba = background_for_jump
        frames.append(current_frame_canvas_rgba.copy())
    return frames

# --- User Image Transformation Helpers ---

def create_cartoon_effect(pil_image,
                           blur_radius=0.5,
                           posterize_bits=4,
                           edge_threshold=70,
                           line_thickness=1):
    """
    Applies an enhanced cartoon effect to a PIL Image.
    Returns an RGBA PIL Image.
    """
    try:
        print(f"Cartoon effect params: blur={blur_radius}, posterize={posterize_bits}, edge_thresh={edge_threshold}, line_thick={line_thickness}")
        img_rgba_input = pil_image.convert("RGBA")
        original_alpha = img_rgba_input.split()[-1]
        img_rgb = img_rgba_input.convert("RGB")

        if blur_radius > 0:
            print("Applying Gaussian Blur...")
            img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        print("Applying Posterization...")
        posterized_rgb = ImageOps.posterize(img_rgb, posterize_bits)

        print("Detecting edges...")
        edges_l = posterized_rgb.filter(ImageFilter.FIND_EDGES).convert("L")

        print(f"Thresholding edges at {edge_threshold}...")
        enhanced_edges_l = edges_l.point(lambda x: 255 if x > edge_threshold else 0, mode='1').convert("L")

        if line_thickness > 0:
            filter_size = 2 * line_thickness + 1
            print(f"Thickening edges with MaxFilter size {filter_size}...")
            final_edges_l = enhanced_edges_l.filter(ImageFilter.MaxFilter(filter_size))
        else:
            final_edges_l = enhanced_edges_l

        print("Preparing base cartoon RGBA...")
        base_cartoon_rgba = posterized_rgb.convert("RGBA")
        base_cartoon_rgba.putalpha(original_alpha)

        print("Creating lines overlay...")
        lines_overlay_rgba = Image.new("RGBA", img_rgba_input.size, (0, 0, 0, 0))
        black_color_rgb_for_lines = Image.new("RGB", img_rgba_input.size, (0, 0, 0))
        lines_overlay_rgba.paste(black_color_rgb_for_lines, (0, 0), mask=final_edges_l)

        print("Compositing lines onto base...")
        cartoon_image_final = Image.alpha_composite(base_cartoon_rgba, lines_overlay_rgba)
        
        print("Cartoon effect applied successfully.")
        return cartoon_image_final.convert("RGBA")

    except Exception as e:
        print(f"!!! ERROR IN create_cartoon_effect (enhanced) !!!: {e}")
        traceback.print_exc()
        return pil_image.convert("RGBA")


def apply_zoom_center_effect(pil_image, zoom_factor=1.3, output_size=None):
    img_rgba = pil_image.convert("RGBA")
    width, height = img_rgba.size
    if output_size is None: output_size = (width, height)
    crop_width, crop_height = int(width / zoom_factor), int(height / zoom_factor)
    if crop_width < 1 or crop_height < 1: return img_rgba
    crop_x, crop_y = (width - crop_width) // 2, (height - crop_height) // 2
    cropped_img = img_rgba.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    return cropped_img.resize(output_size, Image.Resampling.LANCZOS)

def crop_to_circle(pil_image):
    img_rgba = pil_image.convert("RGBA")
    width, height = img_rgba.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)
    img_rgba.putalpha(mask)
    return img_rgba

def process_uploaded_image_to_emoji_impl(image_bytes, effect_type="cartoon", target_size=(128,128)):
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Error opening uploaded image for emoji: {e}"); traceback.print_exc(); return None

    processed_img = img.copy().convert("RGBA")

    if effect_type == "cartoon":
        print("Calling enhanced create_cartoon_effect for user uploaded image...")
        processed_img = create_cartoon_effect(processed_img)
    elif effect_type == "zoom_center":
        print("Applying zoom_center effect to user uploaded image...")
        processed_img = apply_zoom_center_effect(processed_img, zoom_factor=1.3)
    
    if processed_img:
        processed_img = crop_to_circle(processed_img)
        processed_img = processed_img.resize(target_size, Image.Resampling.LANCZOS)
        byte_arr = io.BytesIO()
        processed_img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()
    else:
        print(f"Effect '{effect_type}' failed to return a processed image.")
        return None


def process_uploaded_image_to_gif_impl(image_bytes, animation_type="wiggle", num_frames=12, duration_ms=83):
    animation_frames_rgba = create_simple_animation_frames(image_bytes, animation_type=animation_type, num_frames=num_frames)
    if not animation_frames_rgba:
        print("Failed to create RGBA animation frames for uploaded image GIF."); return None
    gif_bytes_io = io.BytesIO()
    try:
        imageio.mimsave(gif_bytes_io, animation_frames_rgba, format='GIF', duration=duration_ms, subrectangles=True, palettesize=128, loop=0)
        return gif_bytes_io.getvalue()
    except Exception as e:
        print(f"Error saving GIF from uploaded image: {e}"); traceback.print_exc(); return None

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
def index(): return render_template('index.html')

@app.route('/generate-meme', methods=['POST'])
def generate_static_meme_route():
    if not HF_API_TOKEN:
        return jsonify({"error": "Oops! The meme magic is taking a nap. Please try again later."}), 503
    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    if not user_prompt:
        return jsonify({"error": "Prompt cannot be empty!"}), 400
    user_prompt_lower = user_prompt.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in user_prompt_lower:
            print(f"Blocked static prompt: '{user_prompt}' due to: '{keyword}'")
            return jsonify({"error": "Sensitive content in prompt."}), 400 # CORRECTED RETURN
    enhanced_prompt = f"{user_prompt}, meme style, funny, crypto coin, digital art, high detail, vibrant colors"
    neg_prompt = "nsfw, nude, naked, sexually explicit, violence, gore, disturbing, text, watermark, signature, ugly, deformed, blurry, bad anatomy, multiple images"
    payload = {"inputs": enhanced_prompt, "parameters": { "negative_prompt": neg_prompt }}
    image_bytes = query_hf_static_image(payload)
    if image_bytes:
        increment_meme_count()
        return send_file(io.BytesIO(image_bytes), mimetype='image/jpeg')
    else:
        return jsonify({"error": "Failed to generate image. AI busy. Try again!"}), 503

@app.route('/generate-animated-meme', methods=['POST'])
def generate_animated_meme_route():
    if not HF_API_TOKEN:
        return jsonify({"error": "Animated magic napping (Token missing)."}), 503
    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()
    if not user_prompt:
        return jsonify({"error": "Prompt for animation empty!"}), 400
    user_prompt_lower = user_prompt.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in user_prompt_lower:
            print(f"Blocked anim prompt: '{user_prompt}' due to: '{keyword}'")
            return jsonify({"error": "Sensitive words in anim prompt."}), 400 # CORRECTED RETURN
    cartoon_prompt = f"{user_prompt}, funny cartoon style, simple vector illustration, vibrant colors, clear outlines, meme character, no text, centered"
    neg_prompt_anim = "realistic, photo, 3d, complex, nsfw, violence, text, watermark, blurry, deformed, human, multiple characters, dark"
    payload = {"inputs": cartoon_prompt, "parameters": { "negative_prompt": neg_prompt_anim }}
    base_image_bytes = query_hf_static_image(payload)
    if not base_image_bytes:
        return jsonify({"error": "Failed to generate base image for animation. AI busy."}), 503
    try:
        anim_frames = 12
        anim_dur_ms = 83
        avail_anims = ["wiggle", "rotate", "pulse_scale", "pulse_brightness", "center_jump"]
        anim_type = random.choice(avail_anims)
        print(f"Applying animation: {anim_type}")
        pil_frames = create_simple_animation_frames(base_image_bytes, animation_type=anim_type, num_frames=anim_frames)
        if not pil_frames:
            return jsonify({"error": "Could not create anim frames."}), 500
        gif_io = io.BytesIO()
        imageio.mimsave(gif_io, [f for f in pil_frames if isinstance(f, Image.Image)], format='GIF', duration=anim_dur_ms, palettesize=128, loop=0)
        increment_meme_count()
        return send_file(io.BytesIO(gif_io.getvalue()), mimetype='image/gif', download_name=f'memeking_anim_{anim_type}.gif')
    except Exception as e:
        print(f"Error in GIF animation: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to process animation."}), 500

@app.route('/upload-and-transform', methods=['POST'])
def upload_and_transform_route():
    if 'image_file' not in request.files:
        return jsonify({"error": "No image file part"}), 400
    file = request.files['image_file']
    transform_type = request.form.get('transform_type', 'emoji').lower()
    effect = request.form.get('effect', '').lower()
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            if len(image_bytes) > app.config['MAX_CONTENT_LENGTH']:
                 return jsonify({"error": f"File too large: {MAX_UPLOAD_SIZE_MB}MB max"}), 413

            processed_bytes = None
            mimetype = ''
            dl_name = 'transformed'
            if transform_type == 'emoji':
                valid_effects = ["cartoon", "zoom_center"]
                effect = effect if effect in valid_effects else "cartoon"
                print(f"User img transform: emoji, effect='{effect}'")
                processed_bytes = process_uploaded_image_to_emoji_impl(image_bytes, effect_type=effect)
                mimetype = 'image/png'
                dl_name = f'custom_emoji_{effect}.png'
            elif transform_type == 'gif':
                valid_anims = ["wiggle", "rotate", "pulse_scale", "pulse_brightness", "center_jump"]
                effect = effect if effect in valid_anims else random.choice(valid_anims)
                print(f"User img transform: gif, effect='{effect}'")
                processed_bytes = process_uploaded_image_to_gif_impl(image_bytes, animation_type=effect)
                mimetype = 'image/gif'
                dl_name = f'custom_anim_{effect}.gif'
            else:
                return jsonify({"error": "Invalid transform_type."}), 400

            if processed_bytes:
                print(f"Success: type='{transform_type}', effect='{effect}'")
                return send_file(io.BytesIO(processed_bytes), mimetype=mimetype, download_name=dl_name)
            else:
                print(f"Failure to process bytes: type='{transform_type}', effect='{effect}'")
                return jsonify({"error": f"Failed to process image for {transform_type}."}), 500 # CORRECTED RETURN
        except Exception as e:
            print(f"Error in /upload-and-transform: {e}")
            traceback.print_exc()
            if "cannot identify image file" in str(e).lower():
                return jsonify({"error": "Corrupt/unsupported image."}), 400
            return jsonify({"error": "Internal error processing image."}), 500
    else:
        return jsonify({"error": f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}."}), 400

@app.route('/get-meme-count', methods=['GET'])
def get_meme_count_route():
    with count_lock:
        count = get_meme_count()
        return jsonify({"count": count})

@app.route('/terms')
def terms_page(): return render_template('terms.html')

@app.route('/privacy')
def privacy_page(): return render_template('privacy.html')

if __name__ == '__main__':
    initial_count = 0
    try:
        with count_lock:
            if COUNT_FILE_DIR != "." and os.path.dirname(COUNT_FILE) and not os.path.exists(os.path.dirname(COUNT_FILE)):
                 os.makedirs(os.path.dirname(COUNT_FILE), exist_ok=True)
            with open(COUNT_FILE, "r") as f:
                content = f.read().strip()
                initial_count = int(content) if content else 0
            if not content: # If file was empty or just created
                with open(COUNT_FILE, "w") as fw: fw.write(str(initial_count))
    except (FileNotFoundError, ValueError): # File not found or content is not int
        with open(COUNT_FILE, "w") as f: f.write(str(initial_count))
    except IOError as e:
        print(f"Meme count init error '{COUNT_FILE}': {e}")
        if COUNT_FILE_DIR != ".": # Fallback logic if persistent disk failed
            print(f"Falling back to local count file for initialization.")
            COUNT_FILE = "meme_count.txt"
            try:
                with open(COUNT_FILE, "r") as ff:
                    content = ff.read().strip()
                    initial_count = int(content) if content else 0
                if not content:
                    with open(COUNT_FILE, "w") as ffw: ffw.write(str(initial_count))
            except (FileNotFoundError, ValueError):
                with open(COUNT_FILE, "w") as ffw: ffw.write(str(initial_count))
    print(f"Meme Count Initialized: {initial_count} from '{COUNT_FILE}'")
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)