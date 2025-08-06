#!/usr/bin/env python3

import argparse
import json
import os

from tqdm import tqdm

# Import utility functions
from llm_utils import get_image_files

def main():
    parser = argparse.ArgumentParser(description="Create JSONL metadata file from images and captions")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing images and text files")
    parser.add_argument("--output", type=str, default="metadata.jsonl", help="Output JSONL filename")
    
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        exit(1)

    # Get all image files in the directory
    image_files = get_image_files(args.directory)
    
    if not image_files:
        print(f"No image files found in {args.directory}")
        exit(1)

    output_path = os.path.join(args.directory, args.output)
    
    with open(output_path, 'w') as outfile:
        for img_path in tqdm(image_files, desc="Processing images"):
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img_filename = os.path.basename(img_path)
            
            # Look for corresponding text file
            text_path = os.path.join(os.path.dirname(img_path), f"{base_name}.txt")
            
            if os.path.exists(text_path):
                try:
                    with open(text_path, "r", encoding='utf-8') as file:
                        caption = file.read().strip()
                        
                    entry = {"file_name": img_filename, "prompt": caption}
                    json.dump(entry, outfile)
                    outfile.write('\n')
                    
                except Exception as e:
                    print(f"Error processing {text_path}: {e}")
            else:
                print(f"Warning: No caption file found for {img_filename}")

    print(f"JSONL file created at: {output_path}")

if __name__ == '__main__':
    main()