# mediBuddy
Deepseek-R1 fine-tuned to be an efficient medical assistant chatbot

## Fine-Tuning DeepSeek Model using LoRA

This repository provides a comprehensive guide to fine-tuning the DeepSeek model using LoRA (Low-Rank Adaptation) along with SFTTrainer. The process leverages parameter-efficient fine-tuning and mixed precision training to improve domain-specific performance while reducing memory usage and speeding up training.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Fine-Tuning Steps](#fine-tuning-steps)
- [Mixed Precision Training](#mixed-precision-training)
- [Differences in Model Output](#differences-in-model-output)
- [GPU Information](#gpu-information)
- [Results and Monitoring](#results-and-monitoring)

---

## Overview
This repository details the process of creating an advanced medical assistant chatbot by fine-tuning the DeepSeek-R1 model. By leveraging LoRA for parameter-efficient fine-tuning and employing mixed precision training, MediBuddy is engineered to provide:

## Key Features

- **Domain-Specific Responses:** Accurate, medically informed answers. Tailored outputs for medical queries.
- **Enhanced Contextual Understanding:** Maintains context across complex conversations, multi-turn conversations.
- **Optimized Performance:** Faster training and inference using FP16 mixed precision.
- **Mixed Precision Training:** Utilizes FP16 for improved computational efficiency.
- **Parameter-Efficient Fine-Tuning:** Implements LoRA to adjust only a subset of parameters.
- **High Performance on Modern GPUs:** Fine-tuned on an NVIDIA A100 for optimal throughput.

---

## Medical Application and Disclaimer

MediBuddy has been specifically developed for medical applications, with an emphasis on medical imaging diagnostics and research. The model is fine-tuned on a curated dataset containing medical literature, imaging protocols, and domain-specific queries.

**Disclaimer:**  
This chatbot is intended for research and academic purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

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

## Fine-Tuning Steps
- Load Base Model & Tokenizer
- Apply LoRA Configuration - train the target modules for generations query, key and value proj modules
- Configure Training Parameters with SFT trainer
- Start finetuning by running the trainer

---

## Mixed Precision Training

Mixed precision training uses FP16 (half-precision floating point) arithmetic to reduce memory usage and accelerate computations. In this project, FP16 is enabled by setting fp16=True in the training configuration. Although BF16 is an alternative on some platforms, it has been disabled (bf16=False) to ensure compatibility with our custom unsloth kernels, thereby maintaining training stability.

---

## Differences in Model Output

After fine-tuning, MediBuddy delivers significant improvements over the generic pre-trained model:
|            Aspect            |                    Before Fine-Tuning                   |                      After Fine-Tuning (Deepseek-R1)                      |
|:----------------------------:|:-------------------------------------------------------:|:-------------------------------------------------------------------------:|
| **Response Specificity**     | Generic, broad responses                                | Tailored, medically accurate, and domain-specific responses               |
| **Contextual Understanding** | Limited retention of context in complex queries         | Enhanced context awareness for multi-turn medical dialogs                 |
| **Technical Accuracy**       | Lower precision in handling specialized medical queries | Higher precision with accurate medical terminology and diagnostic details |
| **Coherence**                | Basic coherence without specialized structure           | Improved coherence with structured, relevant medical information          |


---

## GPU Information

The fine-tuning process was executed on an NVIDIA A100 GPU (40GB), selected for its robust support for FP16 mixed precision training and high throughput, which substantially accelerates the training process.

---

## Results and Monitoring

Training progress and performance metrics are tracked using Weights & Biases. For detailed metrics, visualizations, and
performance comparisons, please refer to our [Wandb Dashboard](https://wandb.ai/bg2502_hpml/huggingface/runs/7dfh52g2?nw=nwuserbg2502)

### Train results
<img width="1435" alt="Screenshot 2025-02-27 at 4 46 21 PM" src="https://github.com/user-attachments/assets/2d392dc5-b60f-4360-a247-8fdd7bfc70d5" />

### GPU Power usage
<img width="1436" alt="Screenshot 2025-02-27 at 4 47 34 PM" src="https://github.com/user-attachments/assets/a56dceab-ea41-486e-90bb-11cd5d7eee56" />


