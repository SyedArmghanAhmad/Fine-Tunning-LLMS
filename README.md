# Fine-Tuning LLMs (Gemma & LLaMA 2) with LoRA, QLoRA, and Major Techniques

This repository contains three fine-tuning projects for Large Language Models (LLMs) using various techniques:

1. **Fine-tuning Gemma 2B with a Custom Dataset in Keras using LoRA**
2. **Fine-tuning LLaMA 2 using PEFT and QLoRA**
3. **Fine-tuning Gemma 2B on a Quotes Dataset using Major Fine-Tuning Techniques**

Each project explores different methods to enhance model efficiency while optimizing performance.

## 1️⃣ Fine-Tuning Gemma 2B with a Custom Dataset in Keras using LoRA

### Overview

This project fine-tunes **Google's Gemma 2B** using a **custom dataset** and implements **LoRA (Low-Rank Adaptation)** for efficient parameter updates in Keras.

### Key Features

- Uses **LoRA** to fine-tune specific layers instead of modifying the entire model.
- Implements **Keras-based training pipeline** for easier experimentation.
- Reduces memory requirements, making fine-tuning feasible on consumer-grade GPUs.

### Technologies Used

- **TensorFlow / Keras**
- **Hugging Face Transformers**
- **PEFT (Parameter Efficient Fine-Tuning)**
- **LoRA (Low-Rank Adaptation)**

### Steps Involved

1. Load the **custom dataset** and preprocess it.
2. Initialize **Gemma 2B** model with LoRA adapters.
3. Train the model using **Keras training loops**.
4. Evaluate model performance after fine-tuning.

---

## 2️⃣ Fine-Tuning LLaMA 2 using PEFT and QLoRA

### Overview

This project fine-tunes **Meta's LLaMA 2 model** using **PEFT (Parameter Efficient Fine-Tuning)** and **QLoRA (Quantized LoRA)** to optimize GPU memory usage.

### Key Features

- **QLoRA** reduces memory usage by **quantizing** the model to **4-bit precision**.
- **LoRA adapters** enable fine-tuning with fewer trainable parameters.
- Uses **PEFT** for easy integration of fine-tuning techniques.
- Leverages **bitsandbytes** for 4-bit quantization.

### Technologies Used

- **Hugging Face Transformers**
- **PEFT (Parameter Efficient Fine-Tuning)**
- **QLoRA (Quantized Low-Rank Adaptation)**
- **bitsandbytes (4-bit quantization)**

### Steps Involved

1. Load the **LLaMA 2** model with QLoRA configuration.
2. Set up **4-bit quantization** to optimize memory efficiency.
3. Fine-tune the model using **PEFT LoRA adapters**.
4. Evaluate and compare results before and after fine-tuning.

---

## 3️⃣ Fine-Tuning Gemma 2B on a Quotes Dataset using Major Fine-Tuning Techniques

### Overview

This project fine-tunes **Gemma 2B** on a **dataset of famous quotes**, utilizing multiple fine-tuning techniques to maximize performance.

### Key Features

- Uses **LoRA**, **QLoRA**, and **Full Fine-Tuning** approaches.
- Leverages **SFT (Supervised Fine-Tuning)** to train the model on structured text.
- Implements **dynamic padding** and **efficient tokenization**.
- Supports **gradient accumulation** for improved training efficiency.

### Technologies Used

- **Hugging Face Transformers**
- **PEFT (LoRA & QLoRA)**
- **SFTTrainer (Supervised Fine-Tuning)**
- **Google Gemma 2B Model**

### Steps Involved

1. Load the **quotes dataset** and preprocess it.
2. Experiment with **full fine-tuning, LoRA, and QLoRA**.
3. Train using **SFTTrainer** with optimized hyperparameters.
4. Generate and evaluate model predictions on unseen quotes.

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- BitsAndBytes (for QLoRA)

### Installation

```sh
pip install transformers peft datasets bitsandbytes torch accelerate
```

## Conclusion

These projects demonstrate how different fine-tuning techniques can be applied to LLMs to optimize efficiency and performance. **LoRA, QLoRA, and full fine-tuning** each serve unique purposes, allowing for adaptability based on computational constraints and project requirements.
