#!/usr/bin/env python3
"""
Monitor and check status of GCP Batch training jobs
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcp_batch.gcp_batch_launcher import GCPBatchLauncher


def format_duration(duration):
    """Format duration for display"""
    if not duration:
        return "N/A"

    seconds = duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_timestamp(timestamp):
    """Format timestamp for display"""
    if not timestamp:
        return "N/A"

    # Convert to datetime if needed
    if hasattr(timestamp, 'ToDatetime'):
        dt = timestamp.ToDatetime()
    else:
        dt = timestamp

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_job_table(jobs):
    """Print jobs in a formatted table"""
    if not jobs:
        print("No jobs found")
        return

    # Print header
    print(f"\n{'Job Name':<40} {'State':<15} {'Created':<20} {'Duration':<15}")
    print("-" * 95)

    # Print jobs
    for job in jobs:
        name = job['name'][:38] + '..' if len(job['name']) > 40 else job['name']
        state = job['state']
        created = format_timestamp(job.get('create_time'))
        duration = format_duration(job.get('run_duration'))

        # Color code by state
        state_emoji = {
            'STATE_UNSPECIFIED': '?',
            'QUEUED': '⏳',
            'SCHEDULED': '📅',
            'RUNNING': '▶️',
            'SUCCEEDED': '✓',
            'FAILED': '✗',
            'DELETION_IN_PROGRESS': '🗑️'
        }

        emoji = state_emoji.get(state, ' ')

        print(f"{name:<40} {emoji} {state:<13} {created:<20} {duration:<15}")


def main():
    parser = argparse.ArgumentParser(
        description="Check status of GCP Batch training jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all jobs
  python check_batch_jobs.py

  # List jobs with specific prefix
  python check_batch_jobs.py --prefix flux

  # Get detailed status of a specific job
  python check_batch_jobs.py --job flux-training-20231201-120000

  # Watch jobs (refresh every 30 seconds)
  python check_batch_jobs.py --watch

  # Show only running jobs
  python check_batch_jobs.py --state RUNNING
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
        help="Get detailed status of specific job by name"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        help="Filter jobs by name prefix"
    )

    parser.add_argument(
        "--state",
        type=str,
        choices=['QUEUED', 'SCHEDULED', 'RUNNING', 'SUCCEEDED', 'FAILED'],
        help="Filter jobs by state"
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch jobs (refresh every 30 seconds)"
    )

    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a specific job by name"
    )

    args = parser.parse_args()

    # Initialize launcher
    try:
        launcher = GCPBatchLauncher(args.gcp_config)
    except Exception as e:
        print(f"Error initializing GCP Batch launcher: {e}")
        sys.exit(1)

    # Handle specific job query
    if args.job:
        print(f"Getting status for job: {args.job}")
        status = launcher.get_job_status(args.job)

        if not status:
            print(f"Job not found: {args.job}")
            sys.exit(1)

        print(f"\nJob Details:")
        print(f"  Name: {status['name']}")
        print(f"  State: {status['state']}")
        print(f"  Created: {format_timestamp(status['create_time'])}")
        print(f"  Duration: {format_duration(status.get('run_duration'))}")

        output_bucket = launcher.config['storage']['output_bucket']
        output_prefix = launcher.config['storage'].get('output_prefix', '')

        if output_prefix:
            output_path = f"gs://{output_bucket}/{output_prefix}/{args.job}/"
        else:
            output_path = f"gs://{output_bucket}/{args.job}/"

        print(f"\nOutputs will be saved to:")
        print(f"  {output_path}")

        print(f"\nView logs in Cloud Console:")
        print(f"  https://console.cloud.google.com/batch/jobs/details/{launcher.region}/{args.job}?project={launcher.project_id}")

        sys.exit(0)

    # Handle job deletion
    if args.delete:
        response = input(f"Are you sure you want to delete job '{args.delete}'? [y/N]: ")
        if response.lower() in ['y', 'yes']:
            success = launcher.delete_job(args.delete)
            sys.exit(0 if success else 1)
        else:
            print("Cancelled")
            sys.exit(0)

    # List jobs
    def list_and_print_jobs():
        jobs = launcher.list_jobs(filter_prefix=args.prefix)

        if args.state:
            jobs = [j for j in jobs if j['state'] == args.state]

        print(f"\n{'='*60}")
        print(f"GCP Batch Training Jobs")
        print(f"{'='*60}")
        print(f"Project: {launcher.project_id}")
        print(f"Region: {launcher.region}")

        if args.prefix:
            print(f"Filter: Jobs starting with '{args.prefix}'")

        if args.state:
            print(f"Filter: State = {args.state}")

        print_job_table(jobs)

        # Summary
        if jobs:
            state_counts = {}
            for job in jobs:
                state = job['state']
                state_counts[state] = state_counts.get(state, 0) + 1

            print(f"\nSummary:")
            for state, count in sorted(state_counts.items()):
                print(f"  {state}: {count}")

            # Show commands
            print(f"\nUseful commands:")
            print(f"  Get job details:  python check_batch_jobs.py --job <job-name>")
            print(f"  Download outputs: python download_results.py --job <job-name>")
            print(f"  Delete job:       python check_batch_jobs.py --delete <job-name>")

    # Watch mode
    if args.watch:
        import time

        print("Watch mode enabled (Ctrl+C to exit)")

        try:
            while True:
                # Clear screen (works on Unix-like systems)
                print("\033[2J\033[H", end="")

                list_and_print_jobs()

                print(f"\nRefreshing in 30 seconds... (Ctrl+C to exit)")
                time.sleep(30)

        except KeyboardInterrupt:
            print("\n\nExiting watch mode")
            sys.exit(0)
    else:
        list_and_print_jobs()


if __name__ == '__main__':
    main()
