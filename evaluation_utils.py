#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Union, Tuple, List, Dict
import lpips
import os

class PerceptualSimilarity:
    """
    Perceptual similarity calculator using LPIPS (Learned Perceptual Image Patch Similarity)
    """
    
    def __init__(self, net='alex', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the perceptual similarity calculator
        
        Args:
            net (str): Network to use ('alex', 'vgg', 'squeeze')
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        
        # Standard normalization for ImageNet pre-trained models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray], target_size: tuple = None) -> torch.Tensor:
        """
        Preprocess image for perceptual similarity calculation
        
        Args:
            image: Image path, PIL Image, or numpy array
            target_size: Optional tuple (width, height) to resize to
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path, PIL Image, or numpy array")
        
        # Resize if target size is specified
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1] for LPIPS
        tensor = transforms.ToTensor()(image)
        tensor = tensor * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
        
        return tensor.unsqueeze(0).to(self.device)
    
    def calculate_similarity(self, image1: Union[str, Image.Image, np.ndarray], 
                           image2: Union[str, Image.Image, np.ndarray]) -> float:
        """
        Calculate perceptual similarity between two images using LPIPS
        Lower values indicate higher similarity
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Perceptual distance (lower = more similar)
        """
        with torch.no_grad():
            # Load images to get dimensions
            if isinstance(image1, str):
                img1_pil = Image.open(image1).convert('RGB')
            elif isinstance(image1, np.ndarray):
                img1_pil = Image.fromarray(image1).convert('RGB')
            else:
                img1_pil = image1.convert('RGB')
                
            if isinstance(image2, str):
                img2_pil = Image.open(image2).convert('RGB')
            elif isinstance(image2, np.ndarray):
                img2_pil = Image.fromarray(image2).convert('RGB')
            else:
                img2_pil = image2.convert('RGB')
            
            # Determine target size (use smaller dimensions to avoid upscaling)
            w1, h1 = img1_pil.size
            w2, h2 = img2_pil.size
            target_size = (min(w1, w2), min(h1, h2))
            
            # Preprocess both images to same size
            img1_tensor = self.preprocess_image(image1, target_size)
            img2_tensor = self.preprocess_image(image2, target_size)
            
            # Calculate LPIPS distance
            distance = self.loss_fn(img1_tensor, img2_tensor)
            
            return distance.item()
    
    def calculate_similarity_batch(self, images1: list, images2: list) -> list:
        """
        Calculate perceptual similarity for batches of images
        
        Args:
            images1: List of first images
            images2: List of second images
            
        Returns:
            List of perceptual distances
        """
        if len(images1) != len(images2):
            raise ValueError("Image lists must have the same length")
        
        distances = []
        with torch.no_grad():
            for img1, img2 in zip(images1, images2):
                distance = self.calculate_similarity(img1, img2)
                distances.append(distance)
        
        return distances


def calculate_lpips_similarity(image1: Union[str, Image.Image, np.ndarray], 
                              image2: Union[str, Image.Image, np.ndarray],
                              net: str = 'alex') -> float:
    """
    Convenience function to calculate LPIPS similarity between two images
    
    Args:
        image1: First image
        image2: Second image  
        net: Network to use ('alex', 'vgg', 'squeeze')
        
    Returns:
        Perceptual distance (lower = more similar)
    """
    calculator = PerceptualSimilarity(net=net)
    return calculator.calculate_similarity(image1, image2)


def calculate_mse(image1: Union[str, Image.Image, np.ndarray], 
                  image2: Union[str, Image.Image, np.ndarray]) -> float:
    """
    Calculate Mean Squared Error between two images
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        MSE value (lower = more similar)
    """
    def load_and_convert(img):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        else:
            img = img.convert('RGB')
        return img
    
    img1_pil = load_and_convert(image1)
    img2_pil = load_and_convert(image2)
    
    # Resize to same dimensions (use smaller dimensions to avoid upscaling)
    w1, h1 = img1_pil.size
    w2, h2 = img2_pil.size
    target_size = (min(w1, w2), min(h1, h2))
    
    img1_resized = img1_pil.resize(target_size, Image.Resampling.LANCZOS)
    img2_resized = img2_pil.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to arrays
    img1_array = np.array(img1_resized).astype(np.float32) / 255.0
    img2_array = np.array(img2_resized).astype(np.float32) / 255.0
    
    mse = np.mean((img1_array - img2_array) ** 2)
    return float(mse)


def calculate_ssim(image1: Union[str, Image.Image, np.ndarray], 
                   image2: Union[str, Image.Image, np.ndarray]) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images
    Requires scikit-image
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        SSIM value (higher = more similar, range [-1, 1])
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("scikit-image is required for SSIM calculation. Install with: pip install scikit-image")
    
    def load_and_convert(img):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        else:
            img = img.convert('RGB')
        return img
    
    img1_pil = load_and_convert(image1)
    img2_pil = load_and_convert(image2)
    
    # Resize to same dimensions (use smaller dimensions to avoid upscaling)
    w1, h1 = img1_pil.size
    w2, h2 = img2_pil.size
    target_size = (min(w1, w2), min(h1, h2))
    
    img1_resized = img1_pil.resize(target_size, Image.Resampling.LANCZOS)
    img2_resized = img2_pil.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to arrays
    img1_array = np.array(img1_resized)
    img2_array = np.array(img2_resized)
    
    # Calculate SSIM for multichannel images
    ssim_value = ssim(img1_array, img2_array, multichannel=True, channel_axis=2)
    return float(ssim_value)


def evaluate_image_pairs(reference_images: list, generated_images: list, 
                        metrics: list = ['lpips', 'mse', 'ssim']) -> dict:
    """
    Evaluate multiple image pairs using various metrics
    
    Args:
        reference_images: List of reference image paths/objects
        generated_images: List of generated image paths/objects
        metrics: List of metrics to calculate ('lpips', 'mse', 'ssim')
        
    Returns:
        Dictionary with metric results
    """
    if len(reference_images) != len(generated_images):
        raise ValueError("Reference and generated image lists must have the same length")
    
    results = {metric: [] for metric in metrics}
    
    for ref_img, gen_img in zip(reference_images, generated_images):
        if 'lpips' in metrics:
            lpips_score = calculate_lpips_similarity(ref_img, gen_img)
            results['lpips'].append(lpips_score)
        
        if 'mse' in metrics:
            mse_score = calculate_mse(ref_img, gen_img)
            results['mse'].append(mse_score)
        
        if 'ssim' in metrics:
            ssim_score = calculate_ssim(ref_img, gen_img)
            results['ssim'].append(ssim_score)
    
    # Calculate averages
    summary = {}
    for metric in metrics:
        summary[f'{metric}_mean'] = np.mean(results[metric])
        summary[f'{metric}_std'] = np.std(results[metric])
    
    return {
        'individual_scores': results,
        'summary': summary
    }


def create_comparison_grid(reference_image: Union[str, Image.Image], 
                          generated_images: List[Union[str, Image.Image]], 
                          lora_names: List[str],
                          metrics_data: Dict[str, List[float]],
                          output_path: str,
                          image_size: tuple = (512, 512),
                          text_height: int = 80) -> None:
    """
    Create a comparison grid showing reference image and generated images with metrics
    
    Args:
        reference_image: Reference/input image
        generated_images: List of generated images from different LoRAs
        lora_names: Names of the LoRA models
        metrics_data: Dictionary with metric names and lists of scores
        output_path: Path to save the comparison grid
        image_size: Size to resize images to
        text_height: Height reserved for text above each image
    """
    # Load and resize reference image
    if isinstance(reference_image, str):
        ref_img = Image.open(reference_image).convert('RGB')
    else:
        ref_img = reference_image.convert('RGB')
    ref_img = ref_img.resize(image_size, Image.Resampling.LANCZOS)
    
    # Load and resize generated images
    gen_imgs = []
    for gen_img in generated_images:
        if isinstance(gen_img, str):
            img = Image.open(gen_img).convert('RGB')
        else:
            img = gen_img.convert('RGB')
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        gen_imgs.append(img)
    
    # Calculate grid dimensions
    total_images = 1 + len(gen_imgs)  # reference + generated images
    grid_width = total_images * image_size[0]
    grid_height = image_size[1] + text_height
    
    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(grid)
    
    # Paste reference image
    grid.paste(ref_img, (0, text_height))
    
    # Add "Reference" label
    ref_text = "Reference\n(Input)"
    bbox = draw.textbbox((0, 0), ref_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (image_size[0] - text_width) // 2
    draw.text((text_x, 10), ref_text, fill='black', font=font)
    
    # Paste generated images and add metric labels
    for i, (gen_img, lora_name) in enumerate(zip(gen_imgs, lora_names)):
        x_offset = (i + 1) * image_size[0]
        
        # Paste generated image
        grid.paste(gen_img, (x_offset, text_height))
        
        # Create metric text
        metric_lines = [lora_name]
        
        for metric_name, scores in metrics_data.items():
            if i < len(scores):
                score = scores[i]
                if metric_name == 'lpips':
                    metric_lines.append(f"LPIPS: {score:.3f}")
                elif metric_name == 'mse':
                    metric_lines.append(f"MSE: {score:.4f}")
                elif metric_name == 'ssim':
                    metric_lines.append(f"SSIM: {score:.3f}")
                else:
                    metric_lines.append(f"{metric_name.upper()}: {score:.3f}")
        
        metric_text = '\n'.join(metric_lines)
        
        # Calculate text position (centered above image)
        bbox = draw.textbbox((0, 0), metric_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x_offset + (image_size[0] - text_width) // 2
        
        # Draw metric text
        draw.text((text_x, 5), metric_text, fill='black', font=font)
    
    # Save grid
    grid.save(output_path)
    print(f"Comparison grid saved to: {output_path}")


def create_all_comparison_grids(eval_dir: str, 
                               lora_results: Dict[str, Dict], 
                               output_dir: str,
                               metrics: List[str] = ['lpips', 'mse']) -> None:
    """
    Create comparison grids for all evaluation images
    
    Args:
        eval_dir: Directory containing reference images
        lora_results: Results from run_lora_comparison
        output_dir: Directory to save comparison grids
        metrics: List of metrics to display
    """
    # Create grids output directory
    grids_dir = os.path.join(output_dir, "comparison_grids")
    os.makedirs(grids_dir, exist_ok=True)
    
    # Get LoRA names and find all processed filenames
    lora_names = list(lora_results.keys())
    if not lora_names:
        print("No LoRA results found")
        return
    
    # Get all filenames from the first LoRA's results (they should all have the same images)
    first_lora = lora_names[0]
    if 'filename_metrics' not in lora_results[first_lora]:
        print("No filename metrics found in results")
        return
    
    processed_filenames = list(lora_results[first_lora]['filename_metrics'].keys())
    print(f"Creating comparison grids for {len(processed_filenames)} images")
    
    # Process each processed image
    for base_name in processed_filenames:
        # Find reference image path
        ref_img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_path = os.path.join(eval_dir, f"{base_name}{ext}")
            if os.path.exists(potential_path):
                ref_img_path = potential_path
                break
        
        if not ref_img_path:
            print(f"Reference image not found for {base_name}")
            continue
        
        # Find corresponding generated images for each LoRA
        generated_images = []
        valid_lora_names = []
        
        for lora_name in lora_names:
            lora_output_dir = os.path.join(output_dir, lora_name)
            gen_img_path = os.path.join(lora_output_dir, f"{base_name}_generated.png")
            
            if os.path.exists(gen_img_path):
                generated_images.append(gen_img_path)
                valid_lora_names.append(lora_name)
        
        if not generated_images:
            print(f"No generated images found for {base_name}")
            continue
        
        # Collect metrics for this image across all LoRAs using filename lookup
        metrics_data = {}
        for metric in metrics:
            metric_scores = []
            
            for lora_name in valid_lora_names:
                if (lora_name in lora_results and 
                    'filename_metrics' in lora_results[lora_name] and
                    base_name in lora_results[lora_name]['filename_metrics'] and
                    metric in lora_results[lora_name]['filename_metrics'][base_name]):
                    
                    score = lora_results[lora_name]['filename_metrics'][base_name][metric]
                    metric_scores.append(score)
                else:
                    metric_scores.append(0.0)  # Fallback
            
            metrics_data[metric] = metric_scores
        
        print(f"Creating grid for {base_name} with metrics: {metrics_data}")
        
        # Create comparison grid
        grid_path = os.path.join(grids_dir, f"{base_name}_comparison_grid.png")
        create_comparison_grid(
            reference_image=ref_img_path,
            generated_images=generated_images,
            lora_names=valid_lora_names,
            metrics_data=metrics_data,
            output_path=grid_path
        )


if __name__ == "__main__":
    # Example usage
    print("Perceptual Similarity Evaluation Utils")
    print("Required packages: torch, torchvision, lpips, PIL, numpy")
    print("Optional: scikit-image (for SSIM)")
    
    # Example usage would go here
    pass