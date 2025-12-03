# GCP Batch Training Guide

This guide explains how to run AI Toolkit training jobs on Google Cloud Platform (GCP) Batch for parallel, scalable cloud training.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Monitoring](#monitoring)
- [Cost Management](#cost-management)
- [Troubleshooting](#troubleshooting)

## Overview

GCP Batch integration allows you to:
- **Run multiple trainings in parallel** instead of sequentially
- **Use powerful GPU instances** without local hardware
- **Scale** to many concurrent jobs
- **Automatically save** outputs to cloud storage
- **Monitor** jobs from anywhere

### Architecture

```
Local Machine                    GCP Cloud
    |                                |
    |-- Config Files                 |
    |   (YAML)                       |
    |                                |
    +--> GCP Batch Launcher -------> Batch Job 1 --> GCS Bucket
         (Python Script)              Batch Job 2     (Models)
                                      Batch Job 3
                                      ...
```

## Prerequisites

### 1. GCP Account

- Active GCP project with billing enabled
- Appropriate permissions (Batch Admin, Storage Admin, Compute Admin)

### 2. Local Requirements

```bash
# Install Python dependencies (from root directory)
cd ..
pip install -r requirements.txt
cd gcp_batch

# Install gcloud CLI (if not already installed)
# Visit: https://cloud.google.com/sdk/docs/install
```

### 3. GCP APIs

The following APIs must be enabled (automated by `setup_gcp.py`):
- Batch API (`batch.googleapis.com`)
- Compute Engine API (`compute.googleapis.com`)
- Cloud Storage API (`storage.googleapis.com`)
- Cloud Logging API (`logging.googleapis.com`)

### 4. GPU Quotas

Ensure you have GPU quota in your chosen region. Default configuration uses:
- **GPU Type**: NVIDIA Tesla T4
- **Region**: us-central1

Check quotas at: https://console.cloud.google.com/iam-admin/quotas

## Initial Setup

### Step 1: Authenticate with GCP

```bash
# Login to GCP
gcloud auth login

# Set application default credentials
gcloud auth application-default login

# Set your default project
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Run Setup Script

The setup script automates most configuration:

```bash
# Basic setup
python setup_gcp.py --project YOUR_PROJECT_ID --region us-central1

# Setup with bucket creation
python setup_gcp.py --project YOUR_PROJECT_ID --region us-central1 --create-buckets
```

This will:
- ✓ Verify gcloud installation
- ✓ Check authentication
- ✓ Enable required APIs
- ✓ Create `gcp_config.yaml`
- ✓ Create GCS buckets (if --create-buckets flag used)

### Step 3: Customize Configuration

Edit `gcp_config.yaml` to match your needs:

```yaml
gcp:
  project_id: "your-project-id"
  region: "us-central1"
  zone: "us-central1-a"

compute:
  machine_type: "n1-standard-4"  # 4 vCPUs, 15 GB RAM
  gpu_type: "nvidia-tesla-t4"    # T4, A100, L4
  gpu_count: 1
  disk_size_gb: 100

storage:
  training_data_bucket: "your-project-training-data"
  output_bucket: "your-project-lora-outputs"
  output_prefix: "lora-training-runs"

training:
  max_runtime_seconds: 7200  # 2 hours
  retry_count: 1
```

### Step 4: Verify Setup

```bash
python launch_batch_training.py --verify-only
```

This checks:
- ✓ GCP authentication
- ✓ Bucket existence
- ✓ Configuration validity

## Configuration

### Training Config Files

Your existing training configs work as-is! Example: `configs/flux.yaml`

```yaml
job: extension
config:
  name: flux_training
  process:
    - type: sd_trainer
      training_folder: output
      trigger_word: MY_TRIGGER_WORD

      network:
        type: lora
        linear: 16
        linear_alpha: 16

      datasets:
        - folder_path: datasets/data  # Will be uploaded to GCS
          caption_ext: txt
          resolution: [512, 768, 1024]

      train:
        batch_size: 4
        steps: 2000
        # ... other training params
```

### GCP Batch Configuration

Key settings in `gcp_config.yaml`:

#### Machine Types

| Type | vCPUs | RAM | Use Case |
|------|-------|-----|----------|
| `n1-standard-4` | 4 | 15 GB | Small models |
| `n1-standard-8` | 8 | 30 GB | Medium models |
| `n1-standard-16` | 16 | 60 GB | Large models |

#### GPU Types

| GPU | VRAM | Performance | Cost |
|-----|------|-------------|------|
| `nvidia-tesla-t4` | 16 GB | Good | $ |
| `nvidia-l4` | 24 GB | Better | $$ |
| `nvidia-tesla-a100` | 40 GB | Best | $$$ |

#### Runtime Settings

```yaml
training:
  max_runtime_seconds: 7200  # Job timeout (2 hours)
  retry_count: 1             # Retries on failure
```

Estimate runtime based on:
- Steps: 2000 steps ≈ 30-60 minutes (T4)
- Batch size: Larger = faster but more memory
- Resolution: Higher = slower

## Usage

### Launch Training Jobs

#### Launch All Configs

```bash
# Launch all configs in configs/ directory
python launch_batch_training.py --all
```

#### Launch Specific Configs

```bash
# Single config
python launch_batch_training.py --config configs/flux.yaml

# Multiple configs
python launch_batch_training.py --config configs/style1.yaml configs/style2.yaml

# All configs in a directory
python launch_batch_training.py --config-dir configs/
```

#### Advanced Options

```bash
# Dry run (preview without launching)
python launch_batch_training.py --all --dry-run

# Limit number of jobs
python launch_batch_training.py --all --max-jobs 3

# Use custom GCP config
python launch_batch_training.py --all --gcp-config my_config.yaml
```

### What Happens When You Launch

1. **Config Upload**: Training config → GCS
2. **Job Creation**: Batch job created with:
   - GPU instance provisioned
   - AI Toolkit installed
   - Dependencies installed
3. **Training Execution**: Model training runs
4. **Output Upload**: Results saved to GCS
5. **Cleanup**: Instance terminated

### Workflow Example

```bash
# 1. Setup (one-time)
python setup_gcp.py --project my-project --create-buckets

# 2. Verify configuration
python launch_batch_training.py --verify-only

# 3. Launch training
python launch_batch_training.py --all

# Output:
# Found 3 config file(s):
#   1. flux.yaml
#   2. style_a.yaml
#   3. style_b.yaml
#
# Launch 3 training job(s) to GCP Batch? [y/N]: y
#
# ✓ Successfully launched: flux-20231201-120000
# ✓ Successfully launched: style-a-20231201-120001
# ✓ Successfully launched: style-b-20231201-120002
```

## Monitoring

### Check Job Status

```bash
# List all jobs
python check_batch_jobs.py

# Filter by prefix
python check_batch_jobs.py --prefix flux

# Get specific job details
python check_batch_jobs.py --job flux-training-20231201-120000

# Watch mode (auto-refresh)
python check_batch_jobs.py --watch

# Filter by state
python check_batch_jobs.py --state RUNNING
```

### Job States

- **QUEUED**: Waiting to start
- **SCHEDULED**: Resources allocated
- **RUNNING**: Training in progress
- **SUCCEEDED**: Completed successfully
- **FAILED**: Job failed

### View Logs

#### Cloud Console

```bash
# Get link from check command
python check_batch_jobs.py --job <job-name>
# Opens: https://console.cloud.google.com/batch/jobs/...
```

#### Command Line

```bash
# View logs for a job
gcloud logging read "resource.type=batch_job AND resource.labels.job_id=<job-name>" --limit 50 --format json
```

### Download Results

```bash
# Download specific job outputs
python download_results.py --job flux-training-20231201-120000

# Download to custom directory
python download_results.py --job <job-name> --output ./my_models

# List available files
python download_results.py --job <job-name> --list-only

# Download all completed jobs
python download_results.py --all-completed
```

Output structure:
```
outputs/
└── flux-training-20231201-120000/
    ├── config.yaml
    ├── training.log
    └── output/
        ├── flux_training/
        │   ├── flux_training.safetensors
        │   ├── flux_training_001000.safetensors
        │   └── samples/
        │       ├── 0001.png
        │       └── ...
```

### Manage Jobs

```bash
# Delete a job
python check_batch_jobs.py --delete <job-name>

# Or use gcloud directly
gcloud batch jobs delete <job-name> --location us-central1
```

## Cost Management

### Estimating Costs

GCP Batch costs include:
1. **Compute**: VM instance
2. **GPU**: Accelerator
3. **Storage**: GCS storage and operations
4. **Network**: Egress (minimal)

#### Example: T4 GPU Training (2 hours)

- VM (n1-standard-4): ~$0.19/hr × 2 = $0.38
- GPU (T4): ~$0.35/hr × 2 = $0.70
- Storage (100 GB): ~$0.01
- **Total**: ~$1.09 per job

#### Cost Optimization

1. **Use Preemptible VMs** (70% cheaper, may be interrupted):
   ```yaml
   cost_control:
     use_preemptible: true
   ```

2. **Choose appropriate GPU**:
   - T4: Best value for most tasks
   - L4: Better performance/price for newer workloads
   - A100: Only for large models

3. **Optimize runtime**:
   - Test locally first
   - Set accurate `max_runtime_seconds`
   - Use appropriate `steps` count

4. **Batch jobs wisely**:
   ```yaml
   cost_control:
     max_concurrent_jobs: 5  # Limit parallel jobs
   ```

### Monitoring Costs

- **Billing Dashboard**: https://console.cloud.google.com/billing
- **Set Budget Alerts**: Recommended!
  ```bash
  # Via console: Billing → Budgets & alerts
  ```

## Troubleshooting

### Common Issues

#### 1. "Quota exceeded" Error

**Problem**: Insufficient GPU quota

**Solution**:
```bash
# Check current quotas
gcloud compute regions describe us-central1

# Request quota increase:
# https://console.cloud.google.com/iam-admin/quotas
# Search for: "GPUs (all regions)" or specific GPU type
```

#### 2. "Permission denied" Error

**Problem**: Missing IAM permissions

**Solution**:
```bash
# Grant required roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/batch.jobsEditor"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/storage.admin"
```

#### 3. Job Fails Immediately

**Problem**: Startup script error

**Solution**:
- Check Cloud Logging for detailed errors
- Verify AI Toolkit repo URL is correct
- Check if additional packages are available

#### 4. Out of Memory Error

**Problem**: GPU/RAM insufficient

**Solution**:
- Increase `machine_type` (more RAM)
- Use larger GPU type
- Reduce `batch_size` in training config
- Enable `gradient_checkpointing`

#### 5. Training Takes Too Long

**Problem**: Job timeout

**Solution**:
```yaml
training:
  max_runtime_seconds: 14400  # Increase to 4 hours
```

### Debug Mode

Enable detailed logging:

```yaml
monitoring:
  log_level: "DEBUG"
```

### Getting Help

1. **Check logs**: `python check_batch_jobs.py --job <name>`
2. **GCP Status**: https://status.cloud.google.com/
3. **Documentation**: https://cloud.google.com/batch/docs

## Best Practices

### 1. Test Locally First

```bash
# Test config locally before cloud
python run_experiment.py --config configs/test.yaml
```

### 2. Start Small

```bash
# Test with one job first
python launch_batch_training.py --config configs/test.yaml

# Then scale up
python launch_batch_training.py --all
```

### 3. Organize Configs

```
configs/
├── production/
│   ├── flux_prod.yaml
│   └── style_prod.yaml
└── experiments/
    ├── test_a.yaml
    └── test_b.yaml
```

```bash
# Launch only production
python launch_batch_training.py --config-dir configs/production
```

### 4. Monitor Actively

```bash
# Use watch mode during launches
python check_batch_jobs.py --watch
```

### 5. Clean Up

```bash
# Delete old jobs after downloading
python check_batch_jobs.py --delete old-job-name

# Clean up GCS (be careful!)
gsutil rm -r gs://your-bucket/old-outputs/
```

## Advanced Topics

### Custom Docker Images

For faster startup, build a custom image with dependencies pre-installed:

```dockerfile
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-0

RUN pip install lpips image-gen-aux
RUN git clone https://github.com/ostris/ai-toolkit.git /ai-toolkit

# Configure gcp_config.yaml to use your image
```

### Using Private Datasets

Upload datasets to GCS once:

```bash
gsutil -m cp -r datasets/my_data gs://your-training-bucket/datasets/
```

Update training config:
```yaml
datasets:
  - folder_path: gs://your-training-bucket/datasets/my_data
```

### Notification on Completion

Add to `gcp_batch_launcher.py`:

```python
# Send notification (email, Slack, etc.)
# when job.status.state == 'SUCCEEDED'
```

## Additional Resources

- **GCP Batch Documentation**: https://cloud.google.com/batch/docs
- **Deep Learning VM Images**: https://cloud.google.com/deep-learning-vm/docs/images
- **GPU Machine Types**: https://cloud.google.com/compute/docs/gpus
- **Pricing Calculator**: https://cloud.google.com/products/calculator

---

## Quick Reference

### Essential Commands

```bash
# Setup
python setup_gcp.py --project MY_PROJECT --create-buckets

# Launch
python launch_batch_training.py --all

# Monitor
python check_batch_jobs.py
python check_batch_jobs.py --watch

# Download
python download_results.py --job <job-name>

# Verify
python launch_batch_training.py --verify-only
```

### File Structure

```
ai_training_toolkit/
├── configs/               # Training configurations
│   ├── flux.yaml
│   └── style.yaml
├── gcp_config.yaml       # GCP Batch configuration
├── gcp_batch_launcher.py # Core launcher module
├── launch_batch_training.py  # Launch script
├── check_batch_jobs.py   # Monitoring script
├── download_results.py   # Download utility
├── setup_gcp.py          # Setup automation
└── outputs/              # Downloaded results
```

---

Need help? Check troubleshooting section or file an issue on GitHub.
