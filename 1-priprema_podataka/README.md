# Priprema podataka

This directory contains tools for LLM evaluation and quality assessment for the Croatian language Loomen FAQ system.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Folder Setup](#data-folder-setup)
- [LLM Evaluation Scripts](#llm-evaluation-scripts)
- [Local Model Hosting](#local-model-hosting)

---

## Overview

This module provides comprehensive tools for:
1. **LLM evaluation** on Croatian language Q&A tasks
2. **Automated quality assessment** using LLM-as-judge
3. **Context validation** for RAG systems
4. **Local and cloud-based model testing**

---

## Environment Setup

### Python Virtual Environment

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
# LLM evaluation and API clients
pip install openai
```

### Environment Variables

For OpenRouter API access (used in evaluation scripts):

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

---

## Data Folder Setup

### Required Files (provided via email)

You should have received the following files by email. Check if the `data/` directory is created and place them there.

**Required files:**

1. **loomen_faq.md** - Main Loomen FAQ document in Croatian
   - Place at: `data/loomen_faq.md`
   - This is the knowledge base document used for evaluation

2. **loomen_faq.hard_questions.json** - Test questions with context segments
   - Place at: `data/loomen_faq.hard_questions.json`
   - Contains challenging questions, expected answers, and context segments
   - Used by evaluation scripts to test model performance

**Expected directory structure:**

```
1-priprema_podataka/
├── data/
│   ├── loomen_faq.md                      # Main FAQ document
│   └── loomen_faq.hard_questions.json     # Test questions dataset
├── prompts/
│   └── llm_odgovor_cijeli_dokument.txt    # Prompt template
├── results/                                # Generated evaluation results
├── assets/
│   └── judgment_template.html             # HTML report template
├── evaluate_local_model.py
├── evaluate_openrouter_model.py
├── evaluation_utils.py
├── judge_evaluation.py
├── sanity_check.py
└── README.md
```

### Verify Data Setup

After placing the files, verify they're correctly set up:

```bash
./sanity_check.py
```

This will validate that all context segments in the questions file exist in the FAQ document.

---

## Python Scripts

This directory contains five Python scripts for LLM evaluation. Each script has comprehensive documentation in its header.

### evaluation_utils.py

Shared utility functions for all evaluation scripts. Provides common functionality for loading data, building prompts, saving results, and formatting output.

**Key functions:** `load_faq_document()`, `load_questions()`, `build_full_prompt()`, `save_results()`

### evaluate_local_model.py

Evaluates local LLM models running on llama.cpp server.

**Quick start:**
```bash
# Edit script to set LLAMA_CPP_HOST and LLAMA_CPP_PORT
./evaluate_local_model.py
```

**What it does:** Connects to llama.cpp server, tests model on Croatian Q&A, saves results to `results/evaluation_*.json`

**See script header for:** Configuration options, output format, example workflow

### evaluate_openrouter_model.py

Evaluates cloud-based models via OpenRouter API (Claude, GPT-4, Gemini, etc.).

**Quick start:**
```bash
export OPENROUTER_API_KEY="your-key-here"
./evaluate_openrouter_model.py --model anthropic/claude-3.5-sonnet
```

**What it does:** Tests cloud models on Croatian Q&A, saves results to `results/evaluation_*.json`

**See script header for:** Supported models, command-line options, example workflows

### judge_evaluation.py

Automated quality assessment using LLM-as-judge methodology.

**Quick start:**
```bash
./judge_evaluation.py  # Interactive menu to select evaluation results
```

**What it does:** Scores model responses on 5 criteria (accuracy, completeness, relevance, clarity, grounding), generates JSON results and HTML reports

**See script header for:** Evaluation criteria, context modes, HTML report features, interpretation guide

### sanity_check.py

Validates data integrity by verifying context segments exist in FAQ document.

**Quick start:**
```bash
./sanity_check.py  # Run after setting up data/ folder
```

**What it does:** Uses n-gram matching to verify all context segments, exits with code 0 if pass / 1 if fail

**See script header for:** Matching algorithm, troubleshooting guide, typical workflow

---

## Local Model Hosting

## llama.cpp

Guide for running GPT-OSS 20B

https://github.com/ggml-org/llama.cpp/discussions/15396

### Compilation


```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

```


### Commands

#### GPT-OSS-120b with cpu offload

./build/bin/llama-server -hf ggml-org/gpt-oss-120b-GGUF  --ctx-size 0 --jinja --host 0.0.0.0 --n-cpu-moe 28 --n-gpu-layers 999 -ub 2048 -b 2048 -fa 1


#### GPT-OSS-20b 

./build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF  --ctx-size 0 --jinja --host 0.0.0.0 -ub 2048 -b 2048 -fa 1

#### Gemma3-27b-it-qat

./build/bin/llama-server -hf unsloth/gemma-3-27b-it-qat-GGUF  --ctx-size 16384 --jinja --host 0.0.0.0 -ub 2048 -b 2048 -fa 1

#### Gemma3-12b-it-qat

./build/bin/llama-server -hf unsloth/gemma-3-12b-it-qat-GGUF  --ctx-size 32384 --jinja --host 0.0.0.0 -ub 2048 -b 2048 -fa 1


#### Gemma3-4b-it-qat

./build/bin/llama-server -hf unsloth/gemma-3-4b-it-qat-gguf  --ctx-size 32384 --jinja --host 0.0.0.0 -ub 2048 -b 2048 -fa 1



# CUDA

## What You Should Install

**Recommended: CUDA Toolkit 12.8** - This matches perfectly with your driver 570 for RTX 3090. Check official documentation for newer GPUs.

## Installation Instructions for Ubuntu

Here are the official steps for installing CUDA 12.8 on Ubuntu:

### Step 1: Download and Install the CUDA Repository Keyring

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

(Replace `ubuntu2204` with your Ubuntu version: `ubuntu2004`, `ubuntu2204`, or `ubuntu2404`)

### Step 2: Update and Install CUDA Toolkit 12.8

```bash
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

### Step 3: Set Up Environment Variables

Add these lines to your `~/.bashrc` file:

```bash
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-12.8
```

Then reload your shell:

```bash
source ~/.bashrc
```

### Step 4: Verify Installation

```bash
nvcc --version
```

## Important Notes

- **Don't install `cuda` package** - it may try to upgrade your driver unnecessarily. Install `cuda-toolkit-12-8` specifically.
- Your driver already supports CUDA 12.8, so you won't need any compatibility packages
- The 570 driver release is validated with CUDA 12.x

