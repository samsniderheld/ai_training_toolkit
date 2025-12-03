#!/usr/bin/env python3
import argparse
import os
import sys


# Add ai-toolkit to path
toolkit_path = os.environ.get('AI_TOOLKIT_PATH', 'ai-toolkit')
if os.path.exists(toolkit_path):
    sys.path.insert(0, toolkit_path)

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

# Try to import toolkit
try:
    from toolkit.job import get_job
    TOOLKIT_AVAILABLE = True
    print("AI Toolkit loaded successfully")
except ImportError as e:
    print(f"Warning: AI Toolkit not available: {e}")
    TOOLKIT_AVAILABLE = False
    get_job = None
    

def run_single_config(config_path):
    """Run a single config file"""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return False

    if not TOOLKIT_AVAILABLE:
        print("Error: AI Toolkit not available. Cannot run job.")
        return False

    try:
        print(f"\n{'='*60}")
        print(f"Loading job from: {config_path}")
        print(f"{'='*60}")
        job = get_job(config_path)

        print("Starting training job...")
        job.run()
        print(f"Training completed successfully for: {config_path}")
        return True

    except Exception as e:
        print(f"Training failed for {config_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run training experiment(s) with evaluation")
    parser.add_argument("--config", nargs="?", help="Path to YAML configuration file or directory containing config files")
    parser.add_argument("--config_dir", help="Directory containing multiple YAML config files (alternative to --config)")

    args = parser.parse_args()

    # Determine if we're processing a single config or a directory
    config_input = args.config_dir if args.config_dir else args.config

    if not config_input:
        config_input = "configs/basic_sd3.yaml"  # Default single config

    if not os.path.exists(config_input):
        print(f"Error: Path not found: {config_input}")
        sys.exit(1)

    # Check if it's a directory or a file
    if os.path.isdir(config_input):
        print(f"Processing all config files in directory: {config_input}")

        # Find all YAML config files in the directory
        config_files = []
        for file in os.listdir(config_input):
            if file.endswith(('.yaml', '.yml')):
                config_files.append(os.path.join(config_input, file))

        if not config_files:
            print(f"Error: No YAML config files found in directory: {config_input}")
            sys.exit(1)

        # Sort for consistent ordering
        config_files.sort()

        print(f"Found {len(config_files)} config file(s):")
        for config_file in config_files:
            print(f"  - {os.path.basename(config_file)}")

        # Run each config file
        results = []
        for i, config_file in enumerate(config_files, 1):
            print(f"\n\n{'#'*60}")
            print(f"# Processing config {i}/{len(config_files)}")
            print(f"{'#'*60}")

            success = run_single_config(config_file)
            results.append((config_file, success))

        # Print summary
        print(f"\n\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful

        print(f"Total configs processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed configs:")
            for config_file, success in results:
                if not success:
                    print(f"  - {os.path.basename(config_file)}")

        if failed > 0:
            sys.exit(1)
    else:
        # Single config file
        success = run_single_config(config_input)
        if not success:
            sys.exit(1)
   
if __name__ == '__main__':
    main()
