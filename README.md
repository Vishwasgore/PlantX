Demo Link - https://youtu.be/U2OBgwzetOU
🌱 PlantX – AI-Based Plant Disease Detection System

PlantX is an AI-powered plant disease detection system that helps farmers and agricultural researchers identify plant diseases from leaf images and receive actionable treatment guidance.
The system combines deep learning (CNN), vision-based analysis, and lightweight language models to deliver reliable and explainable results.

📌 Problem Statement

Farmers often struggle to identify plant diseases at an early stage due to lack of expert access and timely diagnosis.
Incorrect diagnosis leads to:

Crop loss

Reduced yield

Excessive pesticide usage

PlantX addresses this problem by providing an AI-based automated disease detection and advisory system that works on standard CPU hardware.

🎯 Project Objectives

Detect plant diseases from leaf images

Classify diseases with high accuracy using a trained CNN model

Handle low-confidence or unknown images intelligently

Generate understandable explanations and treatment suggestions

Design a modular and scalable system suitable for real-world use

🧠 System Architecture Overview

The system works in three intelligent stages:

CNN-based Disease Classification

Vision Model (BLIP) for fallback analysis

LLM (TinyLLaMA) for explanation and advisory

Farmer Image Upload
        ↓
CNN (MobileNetV2-based)
        ↓
Confidence Check
   ┌───────────────┐
   │ High Confidence│ → Disease Prediction → TinyLLaMA → Advice
   └───────────────┘
           │
           ▼
 Low Confidence / Unknown Image
           ↓
        BLIP (Visual Analysis)
           ↓
        TinyLLaMA
           ↓
     Final Diagnosis & Advice

🧪 Dataset Used

PlantVillage Dataset

Crops included:

Tomato

Potato

Bell Pepper

Total classes: 15

Dataset is organized using directory-based labels, where folder names act as class labels.

dataset/
├── Tomato_Early_blight
├── Tomato_Late_blight
├── Potato_healthy
├── Pepper__bell__Bacterial_spot
└── ...

🏗️ Model Training Approach
CNN Model

Framework: TensorFlow / Keras

Base architecture: MobileNetV2 (Pretrained on ImageNet)

Input size: 224 × 224

Output: Softmax probabilities (15 classes)

Training Strategy

Transfer Learning

Base MobileNetV2 layers frozen initially

Custom classification head trained

Fine-tuning

Upper layers unfrozen

Model fine-tuned on plant disease dataset

Data Augmentation

Rotation, zoom, flip, brightness adjustments

Training Time

~8–9 hours on CPU

Final Performance

Validation Accuracy: 92.29%

Top-3 Accuracy: 99.20%

🔁 Confidence-Based Decision Logic

The system does not blindly trust predictions.

If CNN confidence ≥ 70%:
    Use CNN prediction
Else:
    Use BLIP vision model for image understanding


This makes the system more reliable for real-world usage.

👁️ Vision Model – BLIP

Model: BLIP (Bootstrapped Language Image Pretraining)

Purpose:

Analyze images that CNN is unsure about

Generate a textual description of visual symptoms

Runs fully on CPU

Helps handle:

Out-of-distribution images

Poor-quality images

Unseen diseases

🤖 Language Model – TinyLLaMA (via Ollama)

Model: TinyLLaMA

Used for:

Disease explanation

Causes

Treatment suggestions

Preventive measures

Input to LLM:

CNN disease + confidence
OR

BLIP-generated visual description

Runs locally using Ollama

🧩 Project Structure
agriculture_disease_detection/
├── dataset/
├── models/
│   ├── cnn_model.h5
│   ├── best_model.h5
│   ├── class_indices.json
│   └── training_history.png
├── src/
│   ├── train_cnn.py
│   ├── inference_pipeline.py
│   ├── blip_fallback.py
│   ├── llm_advisor.py
│   └── utils.py
├── test_images/
├── results/
├── requirements.txt
└── README.md

▶️ How to Run the Project Locally
1. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

2. Install Dependencies
pip install -r requirements.txt

3. Train CNN Model
python src/train_cnn.py

4. Run Inference
python src/inference_pipeline.py test_images/sample_leaf.jpg

🚀 Key Features

High-accuracy CNN-based classification

Intelligent fallback using vision-language models

Explainable AI outputs (not just labels)

CPU-friendly deployment

Modular and scalable architecture

Suitable for academic evaluation and real-world use

🔮 Future Enhancements

Add more crops and diseases

Integrate real-time camera input

Deploy backend on cloud (AWS/GCP)

Mobile application integration

Multilingual advisory support

📚 Academic Relevance

This project demonstrates:

Practical application of CNNs and transfer learning

Confidence-aware AI system design

Integration of vision models + LLMs

Explainable AI for agriculture

End-to-end ML system engineering

👤 Author

Vishwas Gore
Final Year Computer Engineering Student
GitHub: https://github.com/Vishwasgore
