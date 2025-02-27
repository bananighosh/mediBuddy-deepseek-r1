# mediBuddy
Deepseek-R1 fine-tuned to be an efficient medical assistant chatbot

# Fine-Tuning DeepSeek Model using LoRA

This repository provides a comprehensive guide to fine-tuning the DeepSeek model using LoRA (Low-Rank Adaptation) along with SFTTrainer. The process leverages parameter-efficient fine-tuning and mixed precision training to improve domain-specific performance while reducing memory usage and speeding up training.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Fine-Tuning Steps](#fine-tuning-steps)
- [Mixed Precision Training](#mixed-precision-training)
- [Differences in Model Output](#differences-in-model-output)
- [GPU Information](#gpu-information)
- [How to Run](#how-to-run)
- [Results and Monitoring](#results-and-monitoring)

---

## Overview

The goal of this project is to fine-tune the DeepSeek model, adapting it for domain-specific tasks. By incorporating LoRA modifications and training on a custom dataset, the model produces more accurate and context-aware outputs compared to its pre-trained counterpart.

---

## Prerequisites

- **Python:** 3.8+
- **PyTorch:** Latest version with GPU support
- **Transformers:** Latest release
- **PEFT:** For parameter-efficient fine-tuning
- **Unsloth (fast_lora):** For optimized LoRA kernels
- **Weights & Biases (wandb):** For experiment tracking
- **Other dependencies:** Listed in `requirements.txt`

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/deepseek-finetune.git
cd deepseek-finetune
pip install -r requirements.txt

