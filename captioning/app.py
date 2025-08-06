from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import argparse
from llm_utils import load_system_prompt, encode_image_to_base64, get_caption, get_image_files

app = Flask(__name__)
CORS(app)
IMAGE_DIRECTORY = None

@app.route('/')
def index():
    return render_template('caption_interface.html')

@app.route('/list_images', methods=['GET'])
def list_images():
    if not IMAGE_DIRECTORY or not os.path.exists(IMAGE_DIRECTORY):
        return jsonify({'success': False, 'message': 'Image directory not set or does not exist'})
    
    try:
        images = get_image_files(IMAGE_DIRECTORY)
        images.sort()
        return jsonify({'success': True, 'images': images})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error listing images: {str(e)}'})

@app.route('/get_image', methods=['POST'])
def get_image():
    try:
        image_path = request.json.get('image_path')
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'message': 'Invalid image path'})
        
        import base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine image type
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif file_ext == '.png':
            mime_type = 'image/png'
        elif file_ext == '.webp':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'  # default
        
        return jsonify({
            'success': True, 
            'image_data': f"data:{mime_type};base64,{image_data}"
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading image: {str(e)}'})

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    try:
        image_path = request.json.get('image_path')
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'message': 'Invalid image path'})
        
        system_prompt = load_system_prompt("synthetic_captions.txt")
        base64_image = encode_image_to_base64(image_path)
        caption = f"{TRIGGER_WORD}, {get_caption(base64_image, system_prompt)}"
        
        return jsonify({'success': True, 'caption': caption})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error generating caption: {str(e)}'})

@app.route('/load_caption', methods=['POST'])
def load_caption():
    try:
        image_path = request.json.get('image_path')
        if not image_path:
            return jsonify({'success': False, 'message': 'No image path provided'})
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        caption_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
        
        if os.path.exists(caption_path):
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            return jsonify({'success': True, 'caption': caption})
        else:
            return jsonify({'success': True, 'caption': ''})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading caption: {str(e)}'})

@app.route('/save_caption', methods=['POST'])
def save_caption():
    try:
        image_path = request.json.get('image_path')
        caption = request.json.get('caption', '')
        
        if not image_path:
            return jsonify({'success': False, 'message': 'No image path provided'})
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        caption_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
        
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption.strip())
        
        return jsonify({'success': True, 'message': 'Caption saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error saving caption: {str(e)}'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask Image Caption Interface")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing images")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the Flask app on")
    parser.add_argument("--trigger_word", type=str, default="LGTCH_MOUSE", help="Trigger word to use in captions")
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        exit(1)
    
    IMAGE_DIRECTORY = args.directory
    TRIGGER_WORD = args.trigger_word
    print(f"Starting Flask app with image directory: {IMAGE_DIRECTORY}")
    app.run(debug=True, port=args.port)