"""
GCP Batch Launcher for AI Toolkit Training Jobs
Submits training jobs to Google Cloud Batch for parallel execution
"""

import os
import yaml
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from google.cloud import batch_v1
    from google.cloud import storage
    from google.cloud import logging as cloud_logging
    from google.auth import default as google_auth_default
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("Warning: GCP packages not installed. Install with: pip install -r requirements.txt")


class GCPBatchLauncher:
    """Handles launching AI Toolkit training jobs on GCP Batch"""

    def __init__(self, gcp_config_path: str = "gcp_config.yaml"):
        """
        Initialize the GCP Batch launcher

        Args:
            gcp_config_path: Path to GCP configuration YAML file
        """
        if not GCP_AVAILABLE:
            raise ImportError("GCP packages not installed. Run: pip install google-cloud-batch google-cloud-storage google-auth")

        self.config = self._load_config(gcp_config_path)
        self._validate_config()

        # Initialize GCP clients
        self.project_id = self.config['gcp']['project_id']
        self.region = self.config['gcp']['region']

        self.batch_client = batch_v1.BatchServiceClient()
        self.storage_client = storage.Client(project=self.project_id)

        # Verify GCP authentication
        self._verify_auth()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load GCP configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"GCP config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """Validate required configuration fields"""
        required_fields = {
            'gcp': ['project_id', 'region'],
            'compute': ['machine_type', 'gpu_type', 'gpu_count'],
            'storage': ['training_data_bucket', 'output_bucket']
        }

        for section, fields in required_fields.items():
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(f"Missing required field: {section}.{field}")

    def _verify_auth(self):
        """Verify GCP authentication"""
        try:
            credentials, project = google_auth_default()
            print(f"✓ Authenticated with GCP project: {project or self.project_id}")
        except Exception as e:
            raise RuntimeError(f"GCP authentication failed: {e}\nRun: gcloud auth application-default login")

    def verify_buckets(self) -> bool:
        """Verify that required GCS buckets exist"""
        buckets_to_check = [
            self.config['storage']['training_data_bucket'],
            self.config['storage']['output_bucket']
        ]

        all_exist = True
        for bucket_name in buckets_to_check:
            try:
                bucket = self.storage_client.get_bucket(bucket_name)
                print(f"✓ Bucket exists: gs://{bucket_name}")
            except Exception as e:
                print(f"✗ Bucket not found: gs://{bucket_name}")
                print(f"  Create with: gsutil mb gs://{bucket_name}")
                all_exist = False

        return all_exist

    def upload_config_to_gcs(self, config_path: str, job_name: str) -> str:
        """
        Upload training config to GCS

        Args:
            config_path: Local path to training config YAML
            job_name: Unique job name

        Returns:
            GCS URI of uploaded config
        """
        bucket_name = self.config['storage']['output_bucket']
        bucket = self.storage_client.bucket(bucket_name)

        # Create blob path
        output_prefix = self.config['storage'].get('output_prefix', '')
        if output_prefix:
            blob_name = f"{output_prefix}/{job_name}/config.yaml"
        else:
            blob_name = f"{job_name}/config.yaml"

        blob = bucket.blob(blob_name)

        # Upload file
        blob.upload_from_filename(config_path)
        gcs_uri = f"gs://{bucket_name}/{blob_name}"

        print(f"✓ Uploaded config to: {gcs_uri}")
        return gcs_uri

    def upload_dataset_to_gcs(self, local_dataset_path: str) -> str:
        """
        Upload dataset folder to GCS

        Args:
            local_dataset_path: Local path to dataset directory

        Returns:
            GCS URI of uploaded dataset
        """
        if not os.path.exists(local_dataset_path):
            raise FileNotFoundError(f"Dataset not found: {local_dataset_path}")

        bucket_name = self.config['storage']['training_data_bucket']
        bucket = self.storage_client.bucket(bucket_name)

        # Upload all files in dataset folder
        dataset_name = os.path.basename(local_dataset_path)
        uploaded_files = 0

        for root, _, files in os.walk(local_dataset_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_dataset_path)
                blob_name = f"datasets/{dataset_name}/{relative_path}"

                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file)
                uploaded_files += 1

        gcs_uri = f"gs://{bucket_name}/datasets/{dataset_name}"
        print(f"✓ Uploaded {uploaded_files} files to: {gcs_uri}")
        return gcs_uri

    def create_startup_script(self, config_gcs_uri: str, job_name: str) -> str:
        """
        Create startup script for Batch job

        Args:
            config_gcs_uri: GCS URI of training config
            job_name: Unique job name

        Returns:
            Startup script as string
        """
        output_bucket = self.config['storage']['output_bucket']
        output_prefix = self.config['storage'].get('output_prefix', '')

        if output_prefix:
            output_gcs_path = f"gs://{output_bucket}/{output_prefix}/{job_name}"
        else:
            output_gcs_path = f"gs://{output_bucket}/{job_name}"

        ai_toolkit_repo = self.config['training'].get('ai_toolkit_repo', 'https://github.com/ostris/ai-toolkit.git')
        ai_toolkit_branch = self.config['training'].get('ai_toolkit_branch', 'main')

        additional_packages = self.config.get('advanced', {}).get('additional_pip_packages', [])
        additional_pip = ' '.join(additional_packages)

        env_vars = self.config['training'].get('env_vars', {})
        env_exports = '\n'.join([f'export {k}="{v}"' for k, v in env_vars.items()])

        pre_training_commands = self.config.get('advanced', {}).get('pre_training_commands', [])
        pre_training_script = '\n'.join(pre_training_commands)

        script = f'''#!/bin/bash
set -e

echo "================================"
echo "Starting AI Toolkit Training Job"
echo "Job Name: {job_name}"
echo "================================"

# Set environment variables
{env_exports}

# Install nvidia drivers if needed
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    /opt/deeplearning/install-driver.sh
fi

# Verify GPU
echo "Checking GPU..."
nvidia-smi

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y git wget

# Clone AI Toolkit
echo "Cloning AI Toolkit..."
cd /workspace
git clone {ai_toolkit_repo}
cd ai-toolkit
git checkout {ai_toolkit_branch}

# Install AI Toolkit dependencies
echo "Installing AI Toolkit..."
pip install -r requirements.txt
'''

        if additional_pip:
            script += f'''
# Install additional packages
echo "Installing additional packages..."
pip install {additional_pip}
'''

        if pre_training_script:
            script += f'''
# Run pre-training commands
echo "Running pre-training commands..."
{pre_training_script}
'''

        script += f'''
# Download training config
echo "Downloading config from {config_gcs_uri}..."
gsutil cp {config_gcs_uri} /workspace/config.yaml

# Modify config to point to GCS paths if needed
# (Config should already have correct dataset paths)

# Set AI Toolkit path
export AI_TOOLKIT_PATH=/workspace/ai-toolkit

# Run training
echo "Starting training..."
cd /workspace
python -c "
import sys
sys.path.insert(0, '/workspace/ai-toolkit')
from toolkit.job import get_job

job = get_job('/workspace/config.yaml')
job.run()
print('Training completed successfully!')
"

# Upload outputs to GCS
echo "Uploading outputs to {output_gcs_path}..."

# Upload model outputs
if [ -d "output" ]; then
    gsutil -m cp -r output/* {output_gcs_path}/output/
fi

# Upload logs
if [ -f "training.log" ]; then
    gsutil cp training.log {output_gcs_path}/training.log
fi

echo "================================"
echo "Job completed successfully!"
echo "Outputs saved to: {output_gcs_path}"
echo "================================"
'''

        return script

    def create_batch_job(self, config_path: str, job_name: Optional[str] = None) -> str:
        """
        Create and submit a GCP Batch job

        Args:
            config_path: Path to training configuration YAML
            job_name: Optional custom job name (auto-generated if not provided)

        Returns:
            Job name
        """
        # Generate job name if not provided
        if job_name is None:
            config_basename = os.path.splitext(os.path.basename(config_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"{config_basename}-{timestamp}"

        # Sanitize job name (GCP Batch requirements)
        job_name = job_name.lower().replace('_', '-').replace('.', '-')

        print(f"\n{'='*60}")
        print(f"Creating Batch job: {job_name}")
        print(f"{'='*60}")

        # Upload config to GCS
        config_gcs_uri = self.upload_config_to_gcs(config_path, job_name)

        # Create startup script
        startup_script = self.create_startup_script(config_gcs_uri, job_name)

        # Build Batch job spec
        job = self._build_batch_job_spec(job_name, startup_script)

        # Submit job
        parent = f"projects/{self.project_id}/locations/{self.region}"

        try:
            operation = self.batch_client.create_job(parent=parent, job=job, job_id=job_name)
            print(f"✓ Batch job created: {job_name}")
            print(f"  View in console: https://console.cloud.google.com/batch/jobs?project={self.project_id}")
            return job_name
        except Exception as e:
            raise RuntimeError(f"Failed to create batch job: {e}")

    def _build_batch_job_spec(self, job_name: str, startup_script: str) -> batch_v1.Job:
        """Build GCP Batch job specification"""

        # Runnable: Script to execute
        runnable = batch_v1.Runnable()
        runnable.script = batch_v1.Runnable.Script()
        runnable.script.text = startup_script

        # Task spec
        task = batch_v1.TaskSpec()
        task.runnables = [runnable]
        task.max_retry_count = self.config['training'].get('retry_count', 1)
        task.max_run_duration = f"{self.config['training']['max_runtime_seconds']}s"

        # Environment variables
        env_vars = self.config['training'].get('env_vars', {})
        for key, value in env_vars.items():
            env = batch_v1.Environment.Variable()
            env.name = key
            env.value = value
            task.environment.variables.append(env)

        # Compute resources
        resources = batch_v1.ComputeResource()
        resources.cpu_milli = 4000  # 4 CPUs
        resources.memory_mib = 16384  # 16 GB
        task.compute_resource = resources

        # Task group
        group = batch_v1.TaskGroup()
        group.task_count = 1
        group.task_spec = task

        # Allocation policy (VM configuration)
        policy = batch_v1.AllocationPolicy()

        # Instance configuration
        instance_policy = batch_v1.AllocationPolicy.InstancePolicy()
        instance_policy.machine_type = self.config['compute']['machine_type']

        # GPU accelerator
        accelerator = batch_v1.AllocationPolicy.Accelerator()
        accelerator.type_ = self.config['compute']['gpu_type']
        accelerator.count = self.config['compute']['gpu_count']
        instance_policy.accelerators = [accelerator]

        # Disk
        disk = batch_v1.AllocationPolicy.Disk()
        disk.size_gb = self.config['compute'].get('disk_size_gb', 100)
        instance_policy.boot_disk = disk

        # VM image
        instance_policy_or_template = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instance_policy_or_template.policy = instance_policy
        instance_policy_or_template.install_gpu_drivers = True

        policy.instances = [instance_policy_or_template]

        # Job spec
        job = batch_v1.Job()
        job.task_groups = [group]
        job.allocation_policy = policy

        # Logging
        logs_policy = batch_v1.LogsPolicy()
        logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        job.logs_policy = logs_policy

        return job

    def list_jobs(self, filter_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List Batch jobs

        Args:
            filter_prefix: Optional job name prefix to filter by

        Returns:
            List of job information dictionaries
        """
        parent = f"projects/{self.project_id}/locations/{self.region}"

        try:
            jobs_list = []
            for job in self.batch_client.list_jobs(parent=parent):
                job_name = job.name.split('/')[-1]

                if filter_prefix and not job_name.startswith(filter_prefix):
                    continue

                job_info = {
                    'name': job_name,
                    'state': job.status.state.name,
                    'create_time': job.create_time,
                    'uid': job.uid
                }
                jobs_list.append(job_info)

            return jobs_list
        except Exception as e:
            print(f"Error listing jobs: {e}")
            return []

    def get_job_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific job

        Args:
            job_name: Name of the job

        Returns:
            Job status dictionary or None if not found
        """
        job_path = f"projects/{self.project_id}/locations/{self.region}/jobs/{job_name}"

        try:
            job = self.batch_client.get_job(name=job_path)

            return {
                'name': job_name,
                'state': job.status.state.name,
                'create_time': job.create_time,
                'run_duration': job.status.run_duration,
                'task_count': len(job.task_groups[0].task_spec.runnables) if job.task_groups else 0
            }
        except Exception as e:
            print(f"Error getting job status: {e}")
            return None

    def delete_job(self, job_name: str) -> bool:
        """
        Delete a Batch job

        Args:
            job_name: Name of the job to delete

        Returns:
            True if successful, False otherwise
        """
        job_path = f"projects/{self.project_id}/locations/{self.region}/jobs/{job_name}"

        try:
            operation = self.batch_client.delete_job(name=job_path)
            operation.result()  # Wait for deletion
            print(f"✓ Deleted job: {job_name}")
            return True
        except Exception as e:
            print(f"Error deleting job: {e}")
            return False

    def download_job_outputs(self, job_name: str, local_output_dir: str):
        """
        Download job outputs from GCS

        Args:
            job_name: Name of the job
            local_output_dir: Local directory to download to
        """
        bucket_name = self.config['storage']['output_bucket']
        output_prefix = self.config['storage'].get('output_prefix', '')

        if output_prefix:
            gcs_prefix = f"{output_prefix}/{job_name}/"
        else:
            gcs_prefix = f"{job_name}/"

        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=gcs_prefix)

        os.makedirs(local_output_dir, exist_ok=True)
        downloaded = 0

        for blob in blobs:
            # Get relative path
            relative_path = blob.name[len(gcs_prefix):]
            if not relative_path:
                continue

            local_file = os.path.join(local_output_dir, relative_path)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)

            blob.download_to_filename(local_file)
            downloaded += 1

        print(f"✓ Downloaded {downloaded} files to: {local_output_dir}")
