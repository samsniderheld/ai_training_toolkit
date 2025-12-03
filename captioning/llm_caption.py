
#!/usr/bin/env python3

import argparse
import json
import os

import openai

from pathlib import Path
from PIL import Image 
from tqdm import tqdm

# Import utility functions
from llm_utils import (
    load_system_prompt,
    encode_image_to_base64,
    get_caption,
    get_image_files,
)

# Set your API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def main():
    parser = argparse.ArgumentParser(description="LLM Image Captioning with YAML Configuration")
    parser.add_argument("--input_folder", type=str, help="Path to the input folder containing images (overrides config)")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder for captioned images (overrides config)")
    parser.add_argument("--system_prompt_file", type=str, help="Path to text file containing system prompt (overrides config)")
    parser.add_argument("--trigger_word", type=str,  help="The trigger word to prepend/replace in captions. Use empty string '' to skip trigger word", default="TRGGR_WORD")
    parser.add_argument("--trigger_word_type", type=str, choices=["prepend", "replace"], default="prepend",
                        help="How to apply the trigger word: 'prepend' adds it to the start, 'replace' replaces a word in the caption")
    parser.add_argument("--replace_word", type=str, help="Word to replace when using trigger_word_type='replace'")
    parser.add_argument("--provider", type=str, choices=["openai", "gemini"], default="openai",
                        help="LLM provider to use: 'openai' or 'gemini'")
    parser.add_argument("--model", type=str, help="Model to use for captioning. Default: gpt-4o for OpenAI, gemini-1.5-flash for Gemini")

    args = parser.parse_args()

    # Set default model based on provider if not specified
    if not args.model:
        if args.provider == "openai":
            args.model = "gpt-4o"
        elif args.provider == "gemini":
            args.model = "gemini-2.0-flash-exp"

    # Override with command line arguments if provided
    input_folder = args.input_folder
    output_folder = args.output_folder
    system_prompt_file = args.system_prompt_file 

    # Load system prompt (defaults to default_system_prompt.txt if no file specified)
    system_prompt = load_system_prompt(system_prompt_file)
    if system_prompt_file:
        print(f"Loaded custom system prompt from: {system_prompt_file}")
    else:
        print("Loaded default system prompt from: default_system_prompt.txt")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Input folder: {input_folder}")
    print(f"  Output folder: {output_folder}")
    print()

    folder_path = input_folder
    image_files = get_image_files(folder_path)
    out_text_Files = []

    os.makedirs(output_folder, exist_ok=True)


    for filename in tqdm(image_files, desc="Generating captions", unit="image"):
            image_path = os.path.join(folder_path, filename)
            base = os.path.basename(filename)
            name = os.path.splitext(base)[0]
            # ext = os.path.basename(filename).splitext()[1]

            text_path = os.path.join(args.output_folder, name + ".txt")
            try:
                # For OpenAI: pass base64 encoded image, for Gemini: pass image path
                if args.provider == "openai":
                    image_input = encode_image_to_base64(image_path)
                else:  # gemini
                    image_input = image_path

                caption = get_caption(image_input, system_prompt, args.model, args.provider)

                # Apply trigger word based on the selected mode (only if trigger_word is not empty)
                if args.trigger_word:
                    if args.trigger_word_type == "prepend":
                        caption = f"{args.trigger_word}, {caption}"
                    elif args.trigger_word_type == "replace":
                        # Validate arguments
                        if not args.replace_word:
                            parser.error("--replace_word is required when using --trigger_word_type=replace")
                        # Replace all occurrences of the replace_word with the trigger_word
                        caption = caption.replace(args.replace_word, args.trigger_word)

                print(f"Caption for {filename}: {caption}")
                # Save caption to .txt file
                with open(text_path, 'w') as f:
                    f.write(caption.strip())
                Image.open(image_path).save(os.path.join(args.output_folder, os.path.basename(filename)))

            except Exception as e:
                tqdm.write(f"Error with {filename}: {e}")

    with open(f'{args.output_folder}/metadata.jsonl', 'w') as outfile:
        for img in image_files:
            name = os.path.splitext(img)[0]
            ext = os.path.splitext(img)[1]

            base = os.path.basename(img)
            name = os.path.splitext(base)[0]

            text_path = os.path.join(args.output_folder, name + ".txt")
            with open(text_path, "r") as file:
                caption = file.read()
            entry = {"file_name":f"{name}{ext}", "prompt": caption}
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == '__main__':
    main()