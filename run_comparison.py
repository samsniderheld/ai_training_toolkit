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

# Import FLUX generation
try:
    from generation.flux_pipeline import FluxControlNetGenerators
    FLUX_EVAL = True
    print("FLUX generators loaded successfully")
except ImportError as e:
    print(f"Error: FLUX generators not available: {e}")
    FLUX_EVAL = False

# Define image concatenation function
def concatenate_images_horizontal(*images):
    from PIL import Image
    if not images:
        return None
    
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    concatenated = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in images:
        concatenated.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return concatenated

# Import evaluation utilities
try:
    from evaluation_utils import calculate_lpips_similarity
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


def run_flux_lora_comparison(eval_dir, lora_paths, output_dir=None, metrics=['lpips', 'mse'], max_samples=None, 
                            controlnet_type='depth'):
    """
    Run FLUX LoRA comparison evaluation focusing on horizontal comparison and metrics
    
    Args:
        eval_dir: Directory containing reference images and captions
        lora_paths: List of paths to LoRA model files
        output_dir: Output directory for results
        metrics: List of metrics to calculate
        controlnet_type: Type of ControlNet ('depth', 'canny', 'union', 'upscaler')
    """
    if not FLUX_EVAL:
        print("Error: FLUX generators not available")
        return False
        
    if not EVAL_UTILS:
        print("Error: Evaluation utils not available")
        return False
    
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return False
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(eval_dir, "flux_lora_comparison")
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
    print(f"Testing {len(lora_paths)} LoRA models with FLUX + {controlnet_type} ControlNet")
    
    # Initialize FLUX components
    try:
        from PIL import Image
        
        print(f"Initializing FLUX generator with {controlnet_type} ControlNet...")
        
        # Create FLUX generator with appropriate ControlNet
        if controlnet_type == 'depth':
            generator = FluxControlNetGenerators.depth()
        elif controlnet_type == 'canny':
            generator = FluxControlNetGenerators.canny()
        elif controlnet_type == 'union':
            generator = FluxControlNetGenerators.union()
        elif controlnet_type == 'upscaler':
            generator = FluxControlNetGenerators.upscaler()
        else:
            print(f"Unknown ControlNet type: {controlnet_type}, using depth")
            generator = FluxControlNetGenerators.depth()
        
        generator.load_pipeline()
        print(f"FLUX model loaded. GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Initialize control preprocessing based on type
        if controlnet_type == 'depth' or controlnet_type == "union":
            try:
                from image_gen_aux import DepthPreprocessor
                print("Initializing depth preprocessor...")
                control_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to("cuda")
            except ImportError:
                print("Warning: image_gen_aux not available, using identity preprocessor")
                control_preprocessor = lambda img: [img]
        elif controlnet_type == 'canny':
            import cv2
            control_preprocessor = lambda img: Image.fromarray(cv2.Canny(np.array(img), 100, 200))
        else:
            control_preprocessor = None
        
        # Store results for all LoRAs and generated images
        all_results = {}
        generated_images = {}  # filename -> {'reference': path, lora_name: generated_path}
        
        # Initialize storage for all images
        for caption_file, image_file in caption_files:
            base_name = os.path.splitext(os.path.basename(caption_file))[0]
            generated_images[base_name] = {'reference': image_file}
        
        # Process each LoRA and generate all images with it (much more efficient!)
        for lora_idx, lora_path in enumerate(lora_paths):
            if not os.path.exists(lora_path):
                print(f"Warning: LoRA file not found: {lora_path}")
                continue
                
            lora_name = os.path.splitext(os.path.basename(lora_path))[0]
            print(f"\n--- Processing LoRA {lora_idx + 1}/{len(lora_paths)}: {lora_name} ---")
            
            # Load LoRA once for all images
            print(f"Loading LoRA: {lora_name}")
            generator.load_lora_weights(lora_path, adapter_name="current_lora")
            generator.set_lora_scale(1.0)
            print(f"LoRA loaded. GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Process all images with this LoRA
            for pair_idx, (caption_file, image_file) in enumerate(caption_files):
                print(f"  Processing image {pair_idx + 1}/{len(caption_files)}")
                
                try:
                    # Read caption
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    if not prompt:
                        print(f"    Skipping empty caption file: {caption_file}")
                        continue
                    
                    base_name = os.path.splitext(os.path.basename(caption_file))[0]
                    print(f"    Generating: {base_name}")
                    
                    # Load and process reference image
                    reference_image = Image.open(image_file).resize((1024, 1024)).convert("RGB")
                    
                    # Process control image based on control type
                    if controlnet_type == 'depth' or controlnet_type == 'union' and control_preprocessor:
                        control_image = control_preprocessor(reference_image)[0].convert("RGB")
                    elif controlnet_type == 'canny' and control_preprocessor:
                        control_image = control_preprocessor(reference_image).convert("RGB")
                    else:
                        control_image = reference_image  # Fallback
                    control_scale = 0.8

                    # Generate image
                    if controlnet_type != "union":
                        generated_image = generator.generate_image(
                            prompt=prompt,
                            control_image=control_image,
                            controlnet_conditioning_scale=control_scale,
                            num_inference_steps=50,
                            guidance_scale=3.5,
                            width=1024,
                            height=1024,
                            seed=42  # For reproducibility
                        )
                    else:
                        generated_image = generator.generate_image(
                            prompt=prompt,
                            control_image=control_image,
                            controlnet_conditioning_scale=control_scale,
                            control_mode=2,
                            num_inference_steps=50,
                            guidance_scale=3.5,
                            width=1024,
                            height=1024,
                            seed=42  # For reproducibility
                        )
                    
                    # Resize and save generated image
                    generated_image = generated_image.resize((1024, 1024)).convert("RGB")
                    output_path = os.path.join(output_dir, f"{base_name}_{lora_name}.png")
                    generated_image.save(output_path)
                    generated_images[base_name][lora_name] = output_path
                    
                    print(f"    Saved: {output_path}")
                    
                    # Clear intermediate tensors
                    del control_image, generated_image
                    clear_gpu_memory()
                    
                except Exception as e:
                    print(f"    Error processing {caption_file}: {e}")
                    continue
            
            # Unload LoRA once after processing all images
            print(f"Unloading LoRA: {lora_name}")
            generator.unload_lora_weights()
            print(f"GPU Memory after LoRA cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Calculate metrics first for all LoRA comparisons
        print(f"\n=== CALCULATING METRICS FOR HORIZONTAL COMPARISONS ===")
        lora_names = [os.path.splitext(os.path.basename(lora_path))[0] for lora_path in lora_paths if os.path.exists(lora_path)]
        
        # Calculate metrics for each image and LoRA combination
        image_metrics = {}  # base_name -> {lora_name: {metric: score}}
        
        for base_name, image_paths in generated_images.items():
            if len(image_paths) > 1:  # Has reference + at least one generated image
                print(f"Calculating metrics for: {base_name}")
                image_metrics[base_name] = {}
                
                for lora_name in lora_names:
                    if lora_name in image_paths:
                        image_metrics[base_name][lora_name] = {}
                        
                        # Calculate each metric
                        for metric in metrics:
                            try:
                                if metric == 'lpips':
                                    score = calculate_lpips_similarity(image_paths['reference'], image_paths[lora_name])
                                elif metric == 'mse':
                                    from evaluation_utils import calculate_mse
                                    score = calculate_mse(image_paths['reference'], image_paths[lora_name])
                                elif metric == 'ssim':
                                    from evaluation_utils import calculate_ssim
                                    score = calculate_ssim(image_paths['reference'], image_paths[lora_name])
                                else:
                                    score = 0.0
                                
                                image_metrics[base_name][lora_name][metric] = score
                            except Exception as e:
                                print(f"    Error calculating {metric} for {lora_name}: {e}")
                                image_metrics[base_name][lora_name][metric] = float('nan')

        # Create horizontal comparisons with metric scores
        print(f"\n=== CREATING HORIZONTAL COMPARISONS WITH SCORES ===")
        
        for base_name, image_paths in generated_images.items():
            if len(image_paths) > 1:  # Has reference + at least one generated image
                try:
                    print(f"Creating comparison for: {base_name}")
                    
                    # Prepare data for comparison grid function
                    generated_image_paths = []
                    valid_lora_names = []
                    
                    # Get generated images in LoRA order
                    for lora_name in lora_names:
                        if lora_name in image_paths:
                            generated_image_paths.append(image_paths[lora_name])
                            valid_lora_names.append(lora_name)
                    
                    if generated_image_paths:
                        # Prepare metrics data in the format expected by create_comparison_grid
                        metrics_data = {}
                        for metric in metrics:
                            metric_scores = []
                            for lora_name in valid_lora_names:
                                if (base_name in image_metrics and 
                                    lora_name in image_metrics[base_name] and
                                    metric in image_metrics[base_name][lora_name]):
                                    score = image_metrics[base_name][lora_name][metric]
                                    metric_scores.append(score if not np.isnan(score) else 0.0)
                                else:
                                    metric_scores.append(0.0)
                            metrics_data[metric] = metric_scores
                        
                        # Import and use the comparison grid function
                        from evaluation_utils import create_comparison_grid
                        comparison_path = os.path.join(output_dir, f"{base_name}_horizontal_comparison.png")
                        
                        create_comparison_grid(
                            reference_image=image_paths['reference'],
                            generated_images=generated_image_paths,
                            lora_names=valid_lora_names,
                            metrics_data=metrics_data,
                            output_path=comparison_path,
                            image_size=(512, 512),
                            text_height=100
                        )
                        
                        print(f"  Saved comparison with scores: {comparison_path}")
                        
                except Exception as e:
                    print(f"Error creating horizontal comparison for {base_name}: {e}")
                    continue
        
        # Organize metrics into results structure for JSON output
        print(f"\n=== ORGANIZING FINAL RESULTS ===")
        
        # Initialize results structure for each LoRA
        for lora_name in lora_names:
            all_results[lora_name] = {
                'filename_metrics': {},
                'summary': {f'{metric}_mean': [] for metric in metrics}
            }
            all_results[lora_name]['summary'].update({f'{metric}_std': [] for metric in metrics})
        
        # Transfer calculated metrics to results structure
        for base_name, lora_metrics in image_metrics.items():
            for lora_name, metric_scores in lora_metrics.items():
                all_results[lora_name]['filename_metrics'][base_name] = metric_scores
        
        # Calculate summary statistics for each LoRA
        for lora_name in lora_names:
            for metric in metrics:
                scores = []
                for filename_data in all_results[lora_name]['filename_metrics'].values():
                    if metric in filename_data and not np.isnan(filename_data[metric]):
                        scores.append(filename_data[metric])
                
                if scores:
                    all_results[lora_name]['summary'][f'{metric}_mean'] = np.mean(scores)
                    all_results[lora_name]['summary'][f'{metric}_std'] = np.std(scores)
                else:
                    all_results[lora_name]['summary'][f'{metric}_mean'] = float('nan')
                    all_results[lora_name]['summary'][f'{metric}_std'] = float('nan')
        
        # Clean up control preprocessor
        if control_preprocessor and hasattr(control_preprocessor, 'to'):
            del control_preprocessor
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
        print(f"Horizontal comparison images saved with '_horizontal_comparison.png' suffix")
        
        return True
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare multiple FLUX LoRA models on evaluation dataset (GPU optimized)")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory with images and captions for evaluation")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory containing LoRA model files (.safetensors)")
    parser.add_argument("--output_dir", type=str, help="Custom output directory for comparison results")
    parser.add_argument("--metrics", type=str, nargs='+', default=['lpips', 'mse'], 
                       help="Metrics to calculate (lpips, mse, ssim)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of evaluation samples to process (default: all)")
    parser.add_argument("--controlnet_type", type=str, choices=['depth', 'canny', 'union', 'upscaler'], 
                       default='depth', help="ControlNet type for FLUX models")
    
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
    
    success = run_flux_lora_comparison(args.eval_dir, lora_files, args.output_dir, args.metrics, args.max_samples,
                                      args.controlnet_type)
    if success:
        print("\nComparison completed successfully!")
    else:
        print("\nComparison failed.")


if __name__ == '__main__':
    main()