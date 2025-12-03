#!/usr/bin/env python3
"""
Launch AI Toolkit training jobs on GCP Batch
Supports launching single configs, multiple configs, or all configs in a directory
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcp_batch.gcp_batch_launcher import GCPBatchLauncher


def find_config_files(path: str) -> List[str]:
    """
    Find config files in a path

    Args:
        path: File path or directory path

    Returns:
        List of config file paths
    """
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        config_files = []
        for file in os.listdir(path):
            if file.endswith(('.yaml', '.yml')):
                config_files.append(os.path.join(path, file))
        return sorted(config_files)

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Launch AI Toolkit training jobs on GCP Batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch all configs in configs/ directory
  python launch_batch_training.py --all

  # Launch specific config file
  python launch_batch_training.py --config configs/flux.yaml

  # Launch multiple specific configs
  python launch_batch_training.py --config configs/style1.yaml configs/style2.yaml

  # Launch all configs in a directory
  python launch_batch_training.py --config-dir configs/

  # Use custom GCP config
  python launch_batch_training.py --all --gcp-config my_gcp_config.yaml

  # Dry run (don't actually submit jobs)
  python launch_batch_training.py --all --dry-run
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        help="Path to training config file(s) or directory"
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        help="Directory containing training configs (alternative to --config)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Launch all configs in default configs/ directory"
    )

    parser.add_argument(
        "--gcp-config",
        type=str,
        default="gcp_config.yaml",
        help="Path to GCP configuration file (default: gcp_config.yaml)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be launched without actually submitting jobs"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify GCP setup (don't launch jobs)"
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Maximum number of jobs to launch (useful for testing)"
    )

    args = parser.parse_args()

    # Determine config files to process
    config_files = []

    if args.all:
        # Use default configs directory
        default_configs_dir = "configs"
        if not os.path.exists(default_configs_dir):
            print(f"Error: Default configs directory not found: {default_configs_dir}")
            sys.exit(1)
        config_files = find_config_files(default_configs_dir)

    elif args.config_dir:
        if not os.path.exists(args.config_dir):
            print(f"Error: Config directory not found: {args.config_dir}")
            sys.exit(1)
        config_files = find_config_files(args.config_dir)

    elif args.config:
        for config_path in args.config:
            if not os.path.exists(config_path):
                print(f"Error: Config not found: {config_path}")
                sys.exit(1)
            config_files.extend(find_config_files(config_path))

    else:
        parser.print_help()
        print("\nError: Must specify --config, --config-dir, or --all")
        sys.exit(1)

    if not config_files:
        print("Error: No config files found")
        sys.exit(1)

    # Apply max jobs limit if specified
    if args.max_jobs and len(config_files) > args.max_jobs:
        print(f"Limiting to {args.max_jobs} jobs (found {len(config_files)} configs)")
        config_files = config_files[:args.max_jobs]

    print(f"\n{'='*60}")
    print(f"GCP Batch Training Launcher")
    print(f"{'='*60}")
    print(f"Found {len(config_files)} config file(s):")
    for i, config_file in enumerate(config_files, 1):
        print(f"  {i}. {os.path.basename(config_file)}")
    print()

    if args.dry_run:
        print("DRY RUN MODE - No jobs will be submitted")
        print()

    # Initialize GCP Batch launcher
    try:
        print(f"Initializing GCP Batch launcher...")
        print(f"Using GCP config: {args.gcp_config}")
        launcher = GCPBatchLauncher(args.gcp_config)
        print()
    except Exception as e:
        print(f"Error initializing GCP Batch launcher: {e}")
        print("\nMake sure you have:")
        print("  1. Created and configured gcp_config.yaml")
        print("  2. Authenticated with GCP: gcloud auth application-default login")
        print("  3. Installed dependencies: pip install -r requirements.txt")
        sys.exit(1)

    # Verify GCP setup
    print("Verifying GCP setup...")
    print()

    if not launcher.verify_buckets():
        print("\nError: Required GCS buckets do not exist")
        print("Create them with:")
        print(f"  gsutil mb gs://{launcher.config['storage']['training_data_bucket']}")
        print(f"  gsutil mb gs://{launcher.config['storage']['output_bucket']}")
        sys.exit(1)

    print()

    if args.verify_only:
        print("✓ GCP setup verified successfully!")
        print("\nYou can now launch training jobs with:")
        print("  python launch_batch_training.py --all")
        sys.exit(0)

    # Confirm before launching (unless dry run)
    if not args.dry_run:
        response = input(f"Launch {len(config_files)} training job(s) to GCP Batch? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled")
            sys.exit(0)
        print()

    # Launch jobs
    launched_jobs = []
    failed_jobs = []

    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'#'*60}")
        print(f"# Job {i}/{len(config_files)}: {os.path.basename(config_file)}")
        print(f"{'#'*60}")

        if args.dry_run:
            print(f"[DRY RUN] Would launch: {config_file}")
            continue

        try:
            job_name = launcher.create_batch_job(config_file)
            launched_jobs.append((config_file, job_name))
            print(f"✓ Successfully launched: {job_name}")

        except Exception as e:
            print(f"✗ Failed to launch: {e}")
            failed_jobs.append((config_file, str(e)))

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if args.dry_run:
        print(f"DRY RUN: Would launch {len(config_files)} job(s)")
    else:
        print(f"Successfully launched: {len(launched_jobs)}")
        print(f"Failed: {len(failed_jobs)}")

        if launched_jobs:
            print("\nLaunched jobs:")
            for config_file, job_name in launched_jobs:
                print(f"  ✓ {job_name} ({os.path.basename(config_file)})")

        if failed_jobs:
            print("\nFailed jobs:")
            for config_file, error in failed_jobs:
                print(f"  ✗ {os.path.basename(config_file)}: {error}")

        if launched_jobs:
            print(f"\nMonitor jobs with:")
            print(f"  python check_batch_jobs.py")
            print(f"\nView in GCP Console:")
            print(f"  https://console.cloud.google.com/batch/jobs?project={launcher.project_id}")

    if failed_jobs:
        sys.exit(1)


if __name__ == '__main__':
    main()
