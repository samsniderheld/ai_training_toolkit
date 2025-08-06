#!/usr/bin/env python3
import argparse
import os
import sys
import json
import gc
import torch
import numpy as np

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

# Import evaluation utilities
try:
    from evaluation_utils import calculate_lpips_similarity, create_all_comparison_grids
    EVAL_UTILS = True
    print("Evaluation utils loaded successfully")
except ImportError as e:
    print(f"Warning: Evaluation utils not available: {e}")
    EVAL_UTILS = False


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def run_lora_comparison(eval_dir, lora_paths, output_dir=None, metrics=['lpips', 'mse'], max_samples=None):
    """
    Run comparison evaluation across multiple LoRA models with GPU memory optimization
    
    Args:
        eval_dir: Directory containing reference images and captions
        lora_paths: List of paths to LoRA model files
        output_dir: Output directory for results
        metrics: List of metrics to calculate
    """
    if not SD_EVAL:
        print("Error: SD35WithControlNet not available")
        return False
        
    if not EVAL_UTILS:
        print("Error: Evaluation utils not available")
        return False
    
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return False
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(eval_dir, "lora_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Limit samples if max_samples is specified
    if max_samples and max_samples < len(caption_files):
        caption_files = caption_files[:max_samples]
        print(f"Limited to {max_samples} samples (out of {len(caption_files)} total)")
    
    print(f"Found {len(caption_files)} caption+image pairs")
    print(f"Testing {len(lora_paths)} LoRA models")
    
    # Initialize components once
    try:
        from image_gen_aux import DepthPreprocessor
        from PIL import Image
        
        print("Initializing depth preprocessor...")
        depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to("cuda")
        
        # Initialize base generator once (without LoRAs)
        print("Loading base SD3.5 model with ControlNet...")
        generator = SD35WithControlNet(
            controlnet_path=[
                "stabilityai/stable-diffusion-3.5-large-controlnet-depth"
            ],
            lora_path=None  # No LoRAs initially
        )
        generator.load_pipeline()
        print(f"Base model loaded. GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Store results for all LoRAs
        all_results = {}
        
        # Process each LoRA model sequentially
        for lora_idx, lora_path in enumerate(lora_paths):
            if not os.path.exists(lora_path):
                print(f"Warning: LoRA file not found: {lora_path}")
                continue
                
            lora_name = os.path.splitext(os.path.basename(lora_path))[0]
            print(f"\n--- Processing LoRA {lora_idx + 1}/{len(lora_paths)}: {lora_name} ---")
            
            # Create LoRA-specific output directory
            lora_output_dir = os.path.join(output_dir, lora_name)
            os.makedirs(lora_output_dir, exist_ok=True)
            
            # Load current LoRA into existing pipeline
            print(f"Loading LoRA: {lora_name}")
            generator.pipeline.load_lora_weights(lora_path, adapter_name="current_lora")
            generator.pipeline.set_adapters("current_lora", adapter_weights=[1.0])
            
            print(f"LoRA loaded: {lora_name}")
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Store paths for evaluation (with filenames as keys)
            image_results = {}  # filename -> {'reference': path, 'generated': path}
            
            # Process each caption+image pair for this LoRA
            for pair_idx, (caption_file, image_file) in enumerate(caption_files):
                print(f"Processing pair {pair_idx + 1}/{len(caption_files)}")
                
                try:
                    # Read caption
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    if not prompt:
                        print(f"Skipping empty caption file: {caption_file}")
                        continue
                    
                    base_name = os.path.splitext(os.path.basename(caption_file))[0]
                    output_path = os.path.join(lora_output_dir, f"{base_name}_generated.png")
                    compare_path = os.path.join(lora_output_dir, f"{base_name}_comparison.png")
                    
                    print(f"Generating image for: {base_name}")
                    
                    # Load and process reference image
                    reference_image = Image.open(image_file).resize((1024, 1024)).convert("RGB")
                    
                    # Process control images
                    depth_image = depth_preprocessor(reference_image, invert=True)[0].convert("RGB")
                    control_images = [depth_image]
                    control_scales = [0.8]
                    
                    # Generate image using ControlNet
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
                    
                    # Create comparison image
                    comparison_image = concatenate_images_horizontal(reference_image, generated_image)
                    comparison_image.save(compare_path)
                    
                    # Store for evaluation using base filename as key
                    image_results[base_name] = {
                        'reference': image_file,
                        'generated': output_path
                    }
                    
                    print(f"Saved: {output_path}")
                    
                    # Clear intermediate tensors
                    del depth_image, control_images, generated_image, comparison_image
                    clear_gpu_memory()
                    
                except Exception as e:
                    print(f"Error processing {caption_file}: {e}")
                    continue
            
            # Evaluate this LoRA's performance
            if image_results:
                print(f"\nEvaluating LoRA: {lora_name}")
                try:
                    # Create filename-based metrics
                    filename_metrics = {}
                    summary_metrics = {f'{metric}_mean': [] for metric in metrics}
                    summary_metrics.update({f'{metric}_std': [] for metric in metrics})
                    
                    # Calculate metrics for each image pair
                    for filename, paths in image_results.items():
                        filename_metrics[filename] = {}
                        
                        for metric in metrics:
                            if metric == 'lpips':
                                score = calculate_lpips_similarity(paths['reference'], paths['generated'])
                            elif metric == 'mse':
                                from evaluation_utils import calculate_mse
                                score = calculate_mse(paths['reference'], paths['generated'])
                            elif metric == 'ssim':
                                from evaluation_utils import calculate_ssim
                                score = calculate_ssim(paths['reference'], paths['generated'])
                            else:
                                score = 0.0
                            
                            filename_metrics[filename][metric] = score
                    
                    # Calculate summary statistics
                    for metric in metrics:
                        scores = [filename_metrics[fname][metric] for fname in filename_metrics.keys()]
                        summary_metrics[f'{metric}_mean'] = np.mean(scores)
                        summary_metrics[f'{metric}_std'] = np.std(scores)
                    
                    all_results[lora_name] = {
                        'filename_metrics': filename_metrics,
                        'summary': summary_metrics
                    }
                    
                    # Print summary for this LoRA
                    print(f"Results for {lora_name}:")
                    for metric, value in summary_metrics.items():
                        print(f"  {metric}: {value:.4f}")
                        
                except Exception as e:
                    print(f"Error evaluating LoRA {lora_name}: {e}")
            
            # Create vertical concatenation for this LoRA
            print(f"Creating vertical concatenation for {lora_name}...")
            comparison_files = []
            for root, _, files in os.walk(lora_output_dir):
                for file in files:
                    if file.endswith('_comparison.png'):
                        comparison_files.append(os.path.join(root, file))
            
            if comparison_files:
                comparison_files.sort()
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
                    concat_path = os.path.join(lora_output_dir, f"{lora_name}_all_comparisons.png")
                    concatenated.save(concat_path)
                    print(f"Vertical concatenation saved to: {concat_path}")
                    
                    # Clean up concatenation images
                    del comparison_images, concatenated
            
            # Clear LoRA memory at end of loop iteration  
            print(f"Finished processing {lora_name}. Clearing LoRA memory...")
            generator.clear_lora_memory()
            print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Clean up depth preprocessor
        del depth_preprocessor
        clear_gpu_memory()
        
        # Save overall comparison results
        results_file = os.path.join(output_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n=== FINAL COMPARISON RESULTS ===")
        print("LoRA Performance Summary:")
        
        # Sort results by LPIPS score if available (lower is better)
        if all_results and 'lpips_mean' in next(iter(all_results.values()))['summary']:
            sorted_results = sorted(all_results.items(), 
                                  key=lambda x: x[1]['summary']['lpips_mean'])
            print("(Sorted by LPIPS score - lower is better)")
        else:
            sorted_results = all_results.items()
        
        for lora_name, results in sorted_results:
            print(f"\n{lora_name}:")
            for metric, value in results['summary'].items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"All outputs saved to: {output_dir}")
        
        # Create comparison grids for each input image
        print("\n=== CREATING COMPARISON GRIDS ===")
        try:
            create_all_comparison_grids(eval_dir, all_results, output_dir, metrics)
            print("Comparison grids created successfully!")
        except Exception as e:
            print(f"Error creating comparison grids: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare multiple LoRA models on evaluation dataset (GPU optimized)")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory with images and captions for evaluation")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory containing LoRA model files (.safetensors)")
    parser.add_argument("--output_dir", type=str, help="Custom output directory for comparison results")
    parser.add_argument("--metrics", type=str, nargs='+', default=['lpips', 'mse'], 
                       help="Metrics to calculate (lpips, mse, ssim)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of evaluation samples to process (default: all)")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("Warning: CUDA not available, running on CPU")
    
    # Validate eval directory
    if not os.path.exists(args.eval_dir):
        print(f"Error: Evaluation directory does not exist: {args.eval_dir}")
        return
    
    # Validate LoRA directory and find LoRA files
    if not os.path.exists(args.lora_dir):
        print(f"Error: LoRA directory does not exist: {args.lora_dir}")
        return
    
    # Find all .safetensors files in the LoRA directory
    lora_files = []
    for file in os.listdir(args.lora_dir):
        if file.endswith('.safetensors'):
            lora_path = os.path.join(args.lora_dir, file)
            lora_files.append(lora_path)
    
    if not lora_files:
        print(f"Error: No .safetensors files found in {args.lora_dir}")
        return
    
    # Sort LoRA files for consistent ordering
    lora_files.sort()
    
    print(f"Found {len(lora_files)} LoRA models in {args.lora_dir}:")
    for lora in lora_files:
        print(f"  - {os.path.basename(lora)}")
    
    success = run_lora_comparison(args.eval_dir, lora_files, args.output_dir, args.metrics, args.max_samples)
    if success:
        print("\nComparison completed successfully!")
    else:
        print("\nComparison failed.")


if __name__ == '__main__':
    main()