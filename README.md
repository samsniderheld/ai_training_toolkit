# Finetuning Hub

Complete training pipeline for fine-tuning Stable Diffusion 3.5 models with AI-Toolkit, including automated captioning, training, and evaluation with ControlNet.

## Overview

This directory provides a comprehensive solution for training custom LoRA models on Stable Diffusion 3.5:
- **AI-Toolkit Integration**: Uses the ai-toolkit framework for robust SD3.5 training
- **Automated Captioning**: LLM-based image captioning with OpenAI GPT-4V
- **YAML Configuration**: Easy-to-modify training configurations
- **Integrated Evaluation**: Automatic post-training evaluation with ControlNet
- **Web App Integration**: Direct integration with the SD35WithControlNet web app

## Directory Structure

```
finetuning/
├── run_experiment.py          # Main training & evaluation script
├── captioning/                # Image captioning tools
│   └── llm_caption.py         # OpenAI GPT-4V image captioning
├── configs/                   # YAML training configurations
│   └── basic_sd3.yaml         # SD3.5 LoRA training config
├── datasets/                  # Training data storage
├── evaluation/                # Evaluation images and captions
└── outputs/                   # Training outputs and generated images
```

## Features

### 1. Automated Training Pipeline
- **YAML-based Configuration**: Easy-to-modify training parameters
- **AI-Toolkit Integration**: Leverages the robust ai-toolkit framework
- **Automatic Model Detection**: Finds and uses the latest trained model
- **Post-training Evaluation**: Automatically evaluates trained models

### 2. LLM-based Captioning
- **OpenAI GPT-4V Integration**: High-quality image descriptions
- **Configurable Prompts**: Custom system prompts via text files
- **Batch Processing**: Process entire directories of images
- **Output Organization**: Organized caption and image storage

### 3. ControlNet Evaluation
- **Web App Integration**: Uses the same SD35WithControlNet as the main app
- **Dual ControlNet**: Depth and Canny edge detection
- **Reference Image Processing**: Uses evaluation images as ControlNet input
- **Automatic Output Management**: Organized generated image storage

### 4. Advanced Training Configuration
- **Prodigy Optimizer**: State-of-the-art adaptive optimizer
- **Cosine LR Scheduling**: Smooth learning rate decay
- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Multiple Resolutions**: 512, 768, and 1024px training buckets

## Prerequisites

### System Requirements
- **GPU**: CUDA-compatible GPU with 16GB+ VRAM
- **Python**: 3.8+
- **Storage**: 20GB+ free space for models and datasets

### Required Dependencies
```bash
# AI Toolkit (clone to /content/ai-toolkit or set AI_TOOLKIT_PATH)
git clone https://github.com/ostris/ai-toolkit.git 

# Python dependencies
cd ai-toolkit
pip -r install requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

1. **Prepare your dataset**:
```bash
# Create captioned dataset
python captioning/llm_caption.py \
  --input_folder /path/to/images \
  --output_folder datasets/my_dataset
```

2. **Run complete training + evaluation**:
```bash
python run_experiment.py configs/basic_sd3.yaml
```

### Detailed Workflows

#### 1. Image Captioning

Generate captions for your training images:

```bash
# Basic captioning
python captioning/llm_caption.py \
  --input_folder datasets/raw_images \
  --output_folder datasets/captioned_data

# Custom system prompt
python captioning/llm_caption.py \
  --input_folder datasets/raw_images \
  --output_folder datasets/captioned_data \
  --system_prompt_file prompts/product_prompt.txt
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

