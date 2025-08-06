#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def get_image_files(root_dir):
    """Recursively find all image files in a directory"""
    image_files = []
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_files.append(os.path.join(folder, file))
    return image_files

def main():
    parser = argparse.ArgumentParser(description="Create evenly spaced dataset split from input directory")
    parser.add_argument("--input_directory", type=str, required=True, help="Path to the input directory containing images and text files")
    parser.add_argument("--output_directory", type=str, required=True, help="Name of the output directory to create")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to extract")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_directory):
        print(f"Error: Input directory '{args.input_directory}' does not exist.")
        exit(1)

    # Get all image files in the input directory
    image_files = get_image_files(args.input_directory)
    image_files.sort()  # Ensure consistent ordering
    
    if not image_files:
        print(f"No image files found in {args.input_directory}")
        exit(1)

    if args.num_samples > len(image_files):
        print(f"Error: Requested {args.num_samples} samples but only {len(image_files)} images found.")
        exit(1)

    # Create output directory
    os.makedirs(args.output_directory, exist_ok=True)
    
    # Calculate evenly spaced indices
    if args.num_samples == 1:
        selected_indices = [len(image_files) // 2]  # Middle image
    else:
        # Evenly space samples across the range
        step = len(image_files) / args.num_samples
        selected_indices = [int(i * step) for i in range(args.num_samples)]
    
    print(f"Selecting {args.num_samples} samples from {len(image_files)} total images")
    print(f"Selected indices: {selected_indices}")
    
    # Copy selected files
    copied_count = 0
    for idx in tqdm(selected_indices, desc="Copying files"):
        img_path = image_files[idx]
        img_filename = os.path.basename(img_path)
        
        # Copy image file
        img_dest = os.path.join(args.output_directory, img_filename)
        shutil.copy2(img_path, img_dest)
        
        # Copy corresponding text file if it exists
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        text_path = os.path.join(os.path.dirname(img_path), f"{base_name}.txt")
        
        if os.path.exists(text_path):
            text_dest = os.path.join(args.output_directory, f"{base_name}.txt")
            shutil.copy2(text_path, text_dest)
        else:
            print(f"Warning: No caption file found for {img_filename}")
        
        copied_count += 1

    print(f"Successfully copied {copied_count} image/caption pairs to {args.output_directory}")

if __name__ == '__main__':
    main()