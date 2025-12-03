#!/usr/bin/env python3
"""
Download training outputs from GCS for completed jobs
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcp_batch.gcp_batch_launcher import GCPBatchLauncher


def main():
    parser = argparse.ArgumentParser(
        description="Download training outputs from GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download outputs for a specific job
  python download_results.py --job flux-training-20231201-120000

  # Download to custom directory
  python download_results.py --job flux-training-20231201-120000 --output ./my_models

  # Download outputs for all completed jobs
  python download_results.py --all-completed

  # List available outputs without downloading
  python download_results.py --job flux-training-20231201-120000 --list-only
        """
    )

    parser.add_argument(
        "--gcp-config",
        type=str,
        default="gcp_config.yaml",
        help="Path to GCP configuration file (default: gcp_config.yaml)"
    )

    parser.add_argument(
        "--job",
        type=str,
        help="Job name to download outputs for"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Local output directory (default: ./outputs/<job-name>)"
    )

    parser.add_argument(
        "--all-completed",
        action="store_true",
        help="Download outputs for all completed jobs"
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available files without downloading"
    )

    args = parser.parse_args()

    if not args.job and not args.all_completed:
        parser.print_help()
        print("\nError: Must specify --job or --all-completed")
        sys.exit(1)

    # Initialize launcher
    try:
        launcher = GCPBatchLauncher(args.gcp_config)
    except Exception as e:
        print(f"Error initializing GCP Batch launcher: {e}")
        sys.exit(1)

    # Get list of jobs to download
    jobs_to_download = []

    if args.all_completed:
        print("Finding completed jobs...")
        all_jobs = launcher.list_jobs()
        jobs_to_download = [j['name'] for j in all_jobs if j['state'] == 'SUCCEEDED']

        if not jobs_to_download:
            print("No completed jobs found")
            sys.exit(0)

        print(f"Found {len(jobs_to_download)} completed job(s):")
        for job_name in jobs_to_download:
            print(f"  - {job_name}")
        print()

    else:
        # Single job
        jobs_to_download = [args.job]

        # Verify job exists
        status = launcher.get_job_status(args.job)
        if not status:
            print(f"Error: Job not found: {args.job}")
            sys.exit(1)

        print(f"Job: {args.job}")
        print(f"State: {status['state']}")

        if status['state'] != 'SUCCEEDED':
            print(f"\nWarning: Job state is {status['state']}, not SUCCEEDED")
            print("Outputs may be incomplete or not available")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                sys.exit(0)

        print()

    # List only mode
    if args.list_only:
        bucket_name = launcher.config['storage']['output_bucket']
        output_prefix = launcher.config['storage'].get('output_prefix', '')

        for job_name in jobs_to_download:
            if output_prefix:
                gcs_prefix = f"{output_prefix}/{job_name}/"
            else:
                gcs_prefix = f"{job_name}/"

            print(f"\nFiles for job '{job_name}':")
            print(f"GCS path: gs://{bucket_name}/{gcs_prefix}")

            bucket = launcher.storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=gcs_prefix))

            if not blobs:
                print("  No files found")
            else:
                for blob in blobs:
                    relative_path = blob.name[len(gcs_prefix):]
                    if relative_path:
                        size_mb = blob.size / (1024 * 1024)
                        print(f"  {relative_path} ({size_mb:.2f} MB)")

        sys.exit(0)

    # Download outputs
    for job_name in jobs_to_download:
        print(f"\n{'='*60}")
        print(f"Downloading: {job_name}")
        print(f"{'='*60}")

        # Determine output directory
        if args.output:
            if len(jobs_to_download) == 1:
                output_dir = args.output
            else:
                # Multiple jobs - create subdirectories
                output_dir = os.path.join(args.output, job_name)
        else:
            output_dir = os.path.join("outputs", job_name)

        try:
            launcher.download_job_outputs(job_name, output_dir)
            print(f"✓ Downloaded to: {output_dir}")

            # Show model files
            model_dir = os.path.join(output_dir, "output")
            if os.path.exists(model_dir):
                print(f"\nModel outputs:")
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith('.safetensors'):
                            rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                            print(f"  {rel_path}")

        except Exception as e:
            print(f"✗ Error downloading outputs: {e}")

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
