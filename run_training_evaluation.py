#!/usr/bin/env python3
import argparse
import os
import sys

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

# Try to import web app generation
try:
    sys.path.insert(0, os.path.join(current_dir, '..', 'web_app'))
    from generation import SD35WithControlNet, concatenate_images_horizontal
    SD_EVAL = True
    print("SD35WithControlNet loaded successfully")
except ImportError as e:
    print(f"Warning: SD35WithControlNet not available: {e}")
    SD_EVAL = False


def run_evaluation(eval_dir, model_path=None, output_dir=None):
    """Run SD3.5 generation on evaluation images and captions using ControlNet"""
    if not SD_EVAL:
        print("Error: SD35WithControlNet not available")
        return False
    
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return False
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(eval_dir, "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize SD3.5 generator like in app.py
        print("Initializing SD35WithControlNet generator...")
        
        # Import required modules
        from image_gen_aux import DepthPreprocessor
        from PIL import Image
        
        # Use custom model path if provided, otherwise use default LoRAs
        if model_path and os.path.exists(model_path):
            print(f"Using custom model: {model_path}")
            lora_paths = [model_path]
        else:
            print("lora model not found. Exiting evaluation.")
            return False
        
        # Initialize generator with ControlNet like in app.py
        generator = SD35WithControlNet(
            controlnet_path=[
                "stabilityai/stable-diffusion-3.5-large-controlnet-depth"
            ],
            lora_path=lora_paths
        )
        generator.load_pipeline()
        generator.set_lora_scales([1.0])  # Default LoRA strengths
        
        # Initialize depth preprocessor like in app.py
        depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to("cuda")
        
        print("Generator initialized and ready!")
        
        # Find all caption files with matching images
        caption_files = []
        for file in os.listdir(eval_dir):
            if file.endswith('.txt'):
                caption_file = os.path.join(eval_dir, file)
                base_name = os.path.splitext(file)[0]
                
                # Check if corresponding image exists
                image_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential_image = os.path.join(eval_dir, base_name + ext)
                    if os.path.exists(potential_image):
                        image_file = potential_image
                        break
                
                if image_file:
                    caption_files.append((caption_file, image_file))
                else:
                    print(f"Warning: No matching image found for {file}")
        
        if not caption_files:
            print(f"No matching caption+image pairs found in {eval_dir}")
            return False
        
        print(f"Found {len(caption_files)} caption+image pairs")
        
        # Process each caption+image pair
        for caption_file, image_file in caption_files:
            try:
                # Read caption
                with open(caption_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                
                if not prompt:
                    print(f"Skipping empty caption file: {caption_file}")
                    continue
                
                base_name = os.path.splitext(os.path.basename(caption_file))[0]
                output_path = os.path.join(output_dir, f"{base_name}_generated.png")
                compare_path = os.path.join(output_dir, f"{base_name}_comparison.png")
                
                print(f"Generating image for: {base_name}")
                print(f"Prompt: {prompt}")
                print(f"Reference image: {os.path.basename(image_file)}")
                
                # Load and process reference image like in app.py
                reference_image = Image.open(image_file).resize((1024, 1024)).convert("RGB")
                
                # Process control images exactly like in app.py
                depth_image = depth_preprocessor(reference_image, invert=True)[0].convert("RGB")
                
                control_images = [depth_image]
                control_scales = [0.8]  # Default scales from web app
                
                # Generate image using ControlNet like in app.py
                generated_image = generator.generate_image(
                    prompt=prompt,
                    control_image=control_images,
                    controlnet_conditioning_scale=control_scales,
                    negative_prompt="Poor quality, weird render, weird pens, misshapen props",
                    num_inference_steps=40,
                    guidance_scale=4.0,
                    width=1024,
                    height=1024
                )
                
                # Save generated image
                generated_image = generated_image.resize((1024, 1024)).convert("RGB")
                generated_image.save(output_path)

                comparison_image = concatenate_images_horizontal(*[reference_image,generated_image])
                comparison_image.save(compare_path)

                print(f"Saved: {output_path}")
                
            except Exception as e:
                print(f"Error processing {caption_file}: {e}")
                continue
        
        # Create vertical concatenation of all comparison images
        print("Creating vertical concatenation of all comparison images...")
        comparison_files = []
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith('_comparison.png'):
                    comparison_files.append(os.path.join(root, file))
        
        if comparison_files:
            comparison_files.sort()  # Sort for consistent ordering
            comparison_images = []
            
            for comp_file in comparison_files:
                try:
                    img = Image.open(comp_file)
                    comparison_images.append(img)
                except Exception as e:
                    print(f"Error loading comparison image {comp_file}: {e}")
            
            if comparison_images:
                # Calculate total height and max width
                total_height = sum(img.height for img in comparison_images)
                max_width = max(img.width for img in comparison_images)
                
                # Create new image with total height
                concatenated = Image.new('RGB', (max_width, total_height))
                
                # Paste images vertically
                y_offset = 0
                for img in comparison_images:
                    concatenated.paste(img, (0, y_offset))
                    y_offset += img.height
                
                # Save concatenated image
                concat_path = os.path.join(output_dir, "all_comparisons_vertical.png")
                concatenated.save(concat_path)
                print(f"Vertical concatenation saved to: {concat_path}")
            else:
                print("No comparison images could be loaded for concatenation")
        else:
            print("No comparison images found for concatenation")
        
        print(f"Evaluation complete! Generated images saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run training experiment with evaluation")
    parser.add_argument("--experiment_name", type=str, default="sd35_test_mx_small", help="Name of the experiment")
    parser.add_argument("--eval_dir", type=str, default="evaluation", help="Directory with images and captions for evaluation")
    parser.add_argument("--model_path", type=str, help="Specific path to safetensors model file")
    parser.add_argument("--output_dir", type=str, help="Custom output directory for generated images")
    
    args = parser.parse_args()
    
    # Use specific model path if provided, otherwise find latest trained model
    if args.model_path:
        if os.path.exists(args.model_path):
            model_path = args.model_path
            print(f"Using specified model: {model_path}")
        else:
            print(f"Error: Specified model path does not exist: {args.model_path}")
            return
    else:
        # Find the latest trained model (assume it's in the training output folder)
        training_folder = f"output/{args.experiment_name}"
        model_path = None
        
        # Look for the latest .safetensors file
        if os.path.exists(training_folder):
            safetensor_files = []
            for root, _, files in os.walk(training_folder):
                for file in files:
                    if file.endswith('.safetensors'):
                        full_path = os.path.join(root, file)
                        safetensor_files.append((full_path, os.path.getmtime(full_path)))
            
            if safetensor_files:
                # Get the most recent model
                model_path = max(safetensor_files, key=lambda x: x[1])[0]
                print(f"Found trained model: {model_path}")
            else:
                print(f"No safetensors files found in {training_folder}")
        else:
            print(f"Training folder does not exist: {training_folder}")
    
    success = run_evaluation(args.eval_dir, model_path, args.output_dir)
    if success:
        print("Evaluation completed successfully!")
    else:
        print("Evaluation failed, but training was successful.")

   
if __name__ == '__main__':
    main()
