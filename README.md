# AI Training Toolkit

Complete training pipeline for fine-tuning diffusion models including FLUX and Stable Diffusion, with automated captioning, training, and evaluation capabilities.

## Overview

This toolkit provides a comprehensive solution for training custom LoRA models on modern diffusion models:
- **Multi-Model Support**: FLUX and Stable Diffusion model training
- **AI-Toolkit Integration**: Uses the ai-toolkit framework for robust training
- **Automated Captioning**: LLM-based image captioning with OpenAI GPT-4V and Google Gemini
- **GCP Batch Training**: Scale training to the cloud with parallel job execution
- **YAML Configuration**: Easy-to-modify training configurations
- **Integrated Evaluation**: Automatic post-training evaluation and comparison
- **FLUX Pipeline**: Native support for FLUX model generation via diffusers

## Directory Structure

```
ai_training_toolkit/
├── run_experiment.py              # Main training script
├── run_training_evaluation.py     # Training evaluation pipeline
├── run_comparison.py              # Model comparison utilities
├── dataset_split.py               # Dataset splitting utilities
├── evaluation_utils.py            # Evaluation helper functions
├── captioning/                    # Image captioning tools
│   ├── llm_caption.py             # LLM image captioning (OpenAI/Gemini)
│   ├── llm_utils.py               # LLM utility functions
│   ├── create_jsonl.py            # JSONL dataset creation
│   ├── app.py                     # Web interface for captioning
│   ├── default_system_prompt.txt  # Default captioning prompt
│   ├── mx_system_prompt.txt       # MX-specific captioning prompt
│   └── templates/                 # Web interface templates
│       └── caption_interface.html
├── configs/                       # YAML training configurations
│   ├── flux.yaml                  # FLUX model training config
│   └── canova_v2_flux.yaml        # FLUX fine-tuning config
├── generation/                    # Image generation pipelines
│   ├── base.py                    # Base generator interface
│   └── flux_pipeline.py           # FLUX diffusers implementation
├── gcp_batch/                     # GCP Batch training integration
│   ├── __init__.py                # Package initialization
│   ├── gcp_batch_launcher.py     # Core GCP Batch launcher
│   ├── launch_batch_training.py  # Launch jobs CLI
│   ├── check_batch_jobs.py       # Monitor jobs CLI
│   ├── download_results.py       # Download outputs CLI
│   ├── setup_gcp.py              # GCP setup automation
│   ├── gcp_config.yaml           # GCP configuration template
│   └── README.md                 # Complete GCP setup guide
├── requirements.txt               # Python dependencies
├── datasets/                      # Training data storage
├── evaluation/                    # Evaluation images and prompts
└── output/                        # Training outputs and generated images
```

## Features

### 1. Multi-Model Training Pipeline
- **FLUX Support**: Native FLUX model training via diffusers
- **YAML-based Configuration**: Easy-to-modify training parameters
- **AI-Toolkit Integration**: Leverages the robust ai-toolkit framework
- **Automatic Model Detection**: Finds and uses the latest trained model
- **Post-training Evaluation**: Automatically evaluates trained models
- **Batch Configuration Processing**: Run multiple configs sequentially or in parallel

### 2. Cloud Training with GCP Batch
- **Parallel Execution**: Train multiple models simultaneously in the cloud
- **GPU Auto-Provisioning**: Automatic GPU instance management
- **Scalable**: Run 5, 10, or 100 training jobs in parallel
- **Cost-Effective**: Pay only for compute time used
- **Cloud Storage Integration**: Automatic upload/download via GCS
- **Job Monitoring**: Track training progress from anywhere
- **See [gcp_batch/README.md](gcp_batch/README.md) for complete setup instructions**

### 3. Advanced Captioning System
- **Multi-Provider Support**: OpenAI GPT-4 and Google Gemini
- **Web Interface**: User-friendly captioning interface via Flask app
- **Configurable Prompts**: Custom system prompts via text files
- **JSONL Export**: Direct dataset creation for training
- **Batch Processing**: Process entire directories of images
- **Trigger Word Management**: Prepend or replace trigger words automatically

### 4. FLUX Generation Pipeline
- **Diffusers Integration**: Native HuggingFace diffusers support
- **Memory Optimization**: CPU offloading and attention slicing
- **LoRA Support**: Load and manage LoRA adapters
- **Flexible Parameters**: Full control over generation settings

### 5. Evaluation & Comparison Tools
- **Model Comparison**: Side-by-side evaluation of different models
- **Training Evaluation**: Automated post-training assessment
- **Dataset Utilities**: Splitting and preprocessing tools
- **Comprehensive Metrics**: Quality and performance evaluation

## Prerequisites

### System Requirements
- **GPU**: CUDA-compatible GPU with 16GB+ VRAM
- **Python**: 3.8+
- **Storage**: 20GB+ free space for models and datasets

### Required Dependencies
```bash
# AI Toolkit (clone to /content/ai-toolkit or set AI_TOOLKIT_PATH)
git clone https://github.com/ostris/ai-toolkit.git

# Python dependencies (for local training)
cd ai-toolkit
pip install -r requirements.txt

# Install this toolkit's dependencies
cd ../ai_training_toolkit
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-api-key-here"        # For OpenAI captioning
export GEMINI_API_KEY="your-api-key-here"        # For Gemini captioning (optional)
```

For GCP Batch training, see [gcp_batch/README.md](gcp_batch/README.md) for additional setup steps.

## Usage

### Quick Start

#### Local Training

1. **Prepare your dataset**:
```bash
# Create captioned dataset (using OpenAI)
python captioning/llm_caption.py \
  --input_folder /path/to/images \
  --output_folder datasets/my_dataset

# Or use Google Gemini
python captioning/llm_caption.py \
  --input_folder /path/to/images \
  --output_folder datasets/my_dataset \
  --provider gemini
```

2. **Run single training job**:
```bash
python run_experiment.py --config configs/flux.yaml
```

3. **Run multiple training jobs sequentially**:
```bash
python run_experiment.py --config-dir configs/
```

#### Cloud Training (GCP Batch)

1. **Setup GCP** (one-time):
```bash
python gcp_batch/setup_gcp.py --project YOUR_PROJECT_ID --create-buckets
```

2. **Launch parallel training**:
```bash
# Launch all configs to cloud
python gcp_batch/launch_batch_training.py --all

# Monitor progress
python gcp_batch/check_batch_jobs.py --watch

# Download results
python gcp_batch/download_results.py --job <job-name>
```

See [gcp_batch/README.md](gcp_batch/README.md) for complete cloud training documentation.

### Detailed Workflows

#### 1. Image Captioning

Generate captions for your training images:

```bash
# Basic captioning with OpenAI
python captioning/llm_caption.py \
  --input_folder datasets/raw_images \
  --output_folder datasets/captioned_data

# Use Google Gemini instead
python captioning/llm_caption.py \
  --input_folder datasets/raw_images \
  --output_folder datasets/captioned_data \
  --provider gemini \
  --model gemini-2.0-flash-exp

# Custom system prompt
python captioning/llm_caption.py \
  --input_folder datasets/raw_images \
  --output_folder datasets/captioned_data \
  --system_prompt_file prompts/product_prompt.txt \
  --trigger_word MYPRODUCT
```

**Example system prompt** (`prompts/product_prompt.txt`):
```
You are an expert product photographer and AI trainer.
Describe images with precise detail focusing on:
- Product features and positioning
- Lighting and composition
- Background and environment
- Style and aesthetic

Always include "LGTCH_MX" when describing Logitech products.
Format: "LGTCH_MX, [detailed description]"
```

#### 2. Training Configuration

Modify `configs/basic_sd3.yaml` for your needs:

```yaml
# Key parameters to adjust
config:
  name: my_custom_model
  process:
    - trigger_word: LGTCH_MX  # Your trigger word
      
      # Dataset configuration
      datasets:
        - folder_path: /path/to/your/captioned/dataset
          caption_dropout_rate: 0.05
          resolution: [512, 768, 1024]
      
      # Training parameters
      train:
        steps: 2000  # Adjust based on dataset size
        batch_size: 1
        lr: 1.0  # Prodigy optimizer learning rate
        optimizer: "Prodigy"
        
      # Sampling configuration
      sample:
        sample_every: 250
        prompts:
          - "LGTCH_MX, modern workspace, professional lighting"
          - "LGTCH_MX, close-up detail shot, clean background"
```

#### 3. Training Execution

```bash
# Run training with evaluation
python run_experiment.py configs/basic_sd3.yaml

# Skip post-training evaluation
python run_experiment.py configs/basic_sd3.yaml --skip-eval

# Custom evaluation directory
python run_experiment.py configs/basic_sd3.yaml --eval-dir custom_evaluation
```

#### 4. Evaluation Setup

Prepare evaluation data in the `evaluation/` directory:
```
evaluation/
├── test_image_1.jpg
├── test_image_1.txt  # "LGTCH_MX, workspace setup, natural lighting"
├── test_image_2.jpg
├── test_image_2.txt  # "LGTCH_MX, close-up detail, studio lighting"
└── ...
```

The evaluation will:
1. Load your trained LoRA model
2. Process each reference image through depth and canny ControlNets
3. Generate new images using the captions and ControlNet guidance
4. Save results to `evaluation/generated_images/`

### Command Line Options

#### run_experiment.py
```bash
python run_experiment.py [config_file] [options]

Arguments:
  config_file               YAML configuration file (default: configs/basic_sd3.yaml)

Options:
  --eval-dir DIR           Evaluation directory (default: evaluation)
  --skip-eval              Skip post-training evaluation
```

#### captioning/llm_caption.py
```bash
python captioning/llm_caption.py [options]

Options:
  --input_folder DIR       Input directory with images
  --output_folder DIR      Output directory for captioned data
  --system_prompt_file     Path to custom system prompt text file
```

## Training Configuration Reference

### Key YAML Sections

#### Network Configuration (LoRA)
```yaml
network:
  type: lora
  linear: 16          # LoRA rank
  linear_alpha: 16    # LoRA alpha
```

#### Training Parameters
```yaml
train:
  batch_size: 1                    # Batch size (usually 1 for SD3.5)
  steps: 2000                     # Total training steps
  lr: 1.0                         # Learning rate (Prodigy adaptive)
  optimizer: "Prodigy"            # Optimizer type
  lr_scheduler: "cosine"          # Learning rate schedule
  noise_offset: 0.1               # Training noise offset
  gradient_checkpointing: true    # Memory optimization
```

#### Dataset Configuration
```yaml
datasets:
  - folder_path: /path/to/data
    caption_ext: txt              # Caption file extension
    caption_dropout_rate: 0.05    # Randomly drop captions 5% of time
    cache_latents_to_disk: true   # Cache for faster training
    resolution: [512, 768, 1024]  # Multi-resolution buckets
```

## Output Structure

After training, you'll find:

```
outputs/
└── my_custom_model/
    ├── models/
    │   ├── my_custom_model_250.safetensors   # Checkpoint at step 250
    │   ├── my_custom_model_500.safetensors   # Checkpoint at step 500
    │   └── my_custom_model_2000.safetensors  # Final model
    ├── samples/                              # Training samples
    └── logs/                                 # Training logs

evaluation/
└── generated_images/
    ├── test_image_1_generated.png           # Generated from test_image_1
    ├── test_image_2_generated.png           # Generated from test_image_2
    └── ...
```

## Integration with Web App

The trained LoRA models can be directly used in the main web application:

1. **Copy trained model**:
```bash
cp outputs/my_model/models/my_model_2000.safetensors \
   ../web_app/models/loras/
```

2. **Update web app configuration** in `../web_app/app.py`:
```python
generator = SD35WithControlNet(
    lora_path=[
        "models/loras/my_model_2000.safetensors",  # Your new model
        "models/loras/existing_model.safetensors"   # Existing models
    ]
)
```

