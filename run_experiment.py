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
    

def main():
    parser = argparse.ArgumentParser(description="Run training experiment with evaluation")
    parser.add_argument("--config", nargs="?", default="configs/basic_sd3.yaml", help="Path to YAML configuration file")

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not TOOLKIT_AVAILABLE:
        print("Error: AI Toolkit not available. Cannot run job.")
        sys.exit(1)
    
    try:
        print(f"Loading job from: {args.config}")
        job = get_job(args.config)
        
        print("Starting training job...")
        job.run()
        print("Training completed successfully!")
        
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
   
if __name__ == '__main__':
    main()
