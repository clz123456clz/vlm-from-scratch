# 🚀 Reproducing **PaliGemma-3b-pt-224** from Scratch  

> **An incredible journey into understanding Vision-Language Models (VLMs)**  

---

## 📘 Overview  

This repository documents my journey of **reproducing a Vision-Language Model (VLM)** from scratch —  
combining **SigLIP** as the Vision Transformer (ViT) encoder and **Gemma** as both the tokenizer and the language decoder.  
The goal was to understand every stage of how **image–text multimodal alignment** works inside modern VLMs like *PaliGemma*.

---

## 🧩 Implemented Components  

### 1️⃣ `modeling_siglip.py`  
- Implements **SigLIP** as a Vision Transformer (ViT).  
- Walks through how images are **sliced into patches using convolutional kernels** and transformed into **token embeddings** through a stack of transformer layers.  
- Reinforces the intuition behind multi-head attention, patch embedding, and positional encoding in vision transformers.

---

### 2️⃣ `processing_paligemma.py`  
- Handles the **image preprocessing pipeline**.  
- Concatenates **prompt tokens** with a special `<image>` token to form the full multimodal input sequence.  
- Shows how text and image tokens are aligned before entering the language model.

---

### 3️⃣ `modeling_gemma.py`  
- Implements the **Gemma language model** from scratch.  
- Revisits the **autoregressive decoder architecture**, **Key-Value Cache (KV-Cache)** for efficient inference, and **RoPE (Rotary Positional Embedding)** for positional encoding.  
- Includes the **projector module** that aligns the visual embeddings from SigLIP with the language embedding space of Gemma.

---

### 4️⃣ Integration & Inference  
- Demonstrates how to **download pretrained weights from Hugging Face**, load them into the self-built architecture, and perform end-to-end inference.  
- Combines **your own test image** and **custom prompts** to generate multimodal text outputs.  
- Provides a complete workflow from preprocessing → model forward pass → text generation.

---

## 🧠 Key Learnings  

- Gained a clear understanding of how **vision and language modalities interact** through token alignment.  
- Learned how **Hugging Face pretrained weights** can be loaded into a self-implemented model for reproducibility and debugging.  
- Reinforced understanding of **Transformer internals**, including attention, embedding projection, and autoregressive decoding.  
- Practiced **from-scratch implementation** skills for research-level architectures while keeping the code compatible with modern libraries.

---



