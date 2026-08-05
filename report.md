# 🐶 Dog Breed Classifier AI — Comprehensive Project Report

---

## 📌 Executive Summary

The **Dog Breed Classifier AI Dashboard** (`DogBreedAI 2.O`) is an end-to-end, deep-learning-powered web application designed to identify dog breeds from images, explain its predictions using spatial saliency heatmaps (**Grad-CAM**), provide rich breed attributes using **Google Gemini AI**, and facilitate multi-turn interactive Q&A through a canine-specialized chatbot assistant (**"Chow"**).

Built using **Python 3.11**, **TensorFlow 2**, **tf-keras**, **Streamlit**, and the new **`google-genai` SDK**, the project combines state-of-the-art Computer Vision transfer learning with Generative AI (LLMs) and Explainable AI (XAI) to create an executive-grade interactive dashboard.

---

## 🛠️ System Architecture & Technology Stack

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|                       Streamlit Executive Glassmorphism UI                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              PREPROCESSING & UTILS                                |
|          - Image EXIF Auto-Rotation (PIL ImageOps)                                |
|          - Image Resizing (224 x 224 px) & Normalization ([0, 1] float32)         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               MODEL INFERENCE LAYER                               |
|   - MobileNetV2 Feature Extractor (TensorFlow Hub: mobilenet_v2_140_224)          |
|   - Dropout Layer (rate=0.2)                                                      |
|   - Dense Classification Layer (120 Output Classes, Softmax Activation)           |
|   - Direct HDF5 Weight Extraction (keras 2/3 cross-version compatibility)         |
+-----------------------------------------------------------------------------------+
                         |                                      |
                         v                                      v
+------------------------------------+  +-------------------------------------------+
|      EXPLAINABLE AI (XAI) LAYER    |  |          GENERATIVE AI LAYER              |
| - Grad-CAM Activation Map          |  | - Google Gemini AI (google-genai SDK)     |
| - Last Conv Layer Gradient Tracking|  | - Structured Breed Intelligence Summaries |
| - OpenCV Heatmap Overlay (JET)     |  | - Grounded Multi-Turn Chat Assistant      |
+------------------------------------+  +-------------------------------------------+
```

### 💻 Key Technologies & Libraries

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit, HTML5, Custom CSS3 | Modern glassmorphism web dashboard |
| **Deep Learning Framework** | TensorFlow 2.21, `tf-keras` | Transfer learning model & inference engine |
| **Pretrained Model Hub** | TensorFlow Hub (`tensorflow-hub`) | Pretrained MobileNetV2 feature extractor |
| **LLM & GenAI Service** | Google Gemini (`google-genai`) | Automated breed attributes & interactive AI chat |
| **Image Processing** | Pillow (`PIL`), OpenCV (`cv2`), NumPy | Image transformation, EXIF orientation, Grad-CAM overlays |
| **Metrics & Evaluation** | Scikit-Learn, Pandas, Matplotlib | Multi-class classification metrics & dataset management |
| **ASGI / Web Server** | Uvicorn, Starlette (`<1.4.0`) | High-performance ASGI server for Streamlit |

---

## 🧠 Deep Learning & Transfer Learning Model

### 1. Model Architecture
The classifier utilizes transfer learning with a fixed feature extractor from **MobileNetV2** (1.40 width multiplier, 224x224 input resolution, trained on ImageNet):

1. **Input Shape**: `(None, 224, 224, 3)` — Float32 normalized RGB tensors `[0.0, 1.0]`.
2. **Feature Extractor Layer (`KerasLayer`)**:
   - **Source**: `https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/5`
   - **Feature Output Dimension**: `1,792` features per image.
   - **Trainable**: `False` (Weights frozen).
3. **Regularization Layer**: `Dropout(rate=0.2)`.
4. **Classification Layer**: `Dense(units=120, activation='softmax')`.

### 2. Dataset Overview
- **Dataset**: Kaggle Dog Breed Identification dataset.
- **Classes**: 120 unique canine breeds (e.g., *Golden Retriever*, *Boxer*, *German Shepherd*, *Chihuahua*, *Pomeranian*).
- **Format**: High-resolution JPEG images mapped via `labels.csv`.

---

## 🔥 Explainable AI (XAI): Grad-CAM Visualizations

To provide visual transparency for AI predictions, the dashboard integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

1. **Feature Extraction**: Tracks activations from the final convolutional feature layer of the MobileNetV2 backbone.
2. **Gradient Calculation**: Computes gradients of the target breed score with respect to feature map activations.
3. **Heatmap Generation**: Applies ReLU activation to positive weighted feature maps, generating a 2D spatial importance grid.
4. **Overlay Rendering**: Normalizes the heatmap, applies OpenCV's `COLORMAP_JET` color spectrum, and overlays it onto the original dog image with 40% alpha transparency.

This allows users to visually verify that the model is focusing on relevant canine features (ears, muzzle, facial structure, coat texture) rather than background noise.

---

## ✨ Google Gemini AI Integration

### 1. Structured Breed Intelligence
When a prediction is generated, the app queries Google's Gemini models (`gemini-flash-latest`, `gemini-2.0-flash`) using structured prompt engineering to instantly retrieve:
- **Origin & Country of Origin**
- **Average Lifespan & Height/Weight Metrics**
- **Temperament & Energy Level**
- **Grooming & Exercise Needs**
- **Key Health Overview & Vulnerabilities**

### 2. Specialized Canine Chat Assistant ("Chow")
An interactive multi-turn chat interface enables users to ask follow-up questions grounded in the predicted breed context.
- **Context Injection**: Automatically injects system prompts restricting answers to canine care, health, diet, and training for the predicted breed.
- **Off-Topic Safety Guardrail**: Gracefully declines non-canine queries (math, code, politics) to keep the assistant domain-focused.

---

## ⚙️ Key Deployment & Cross-Platform Fixes

During the deployment lifecycle to **Streamlit Cloud**, several critical production challenges were identified and engineered for 100% stability:

### 1. Keras 2 / Keras 3 Weight Deserialization Patch (`model_loader.py`)
- **Problem**: When loading `.keras` archives across TensorFlow/Keras versions, `KerasLayer.from_config()` inside standard `load_model()` failed to map frozen MobileNetV2 variables, throwing `Layer 'keras_layer' expected 260 variables, but received 0 variables during loading` and resetting layer weights.
- **Solution**: Implemented direct ZIP/HDF5 weight extraction using `h5py` and `zipfile`. The app instantiates the clean `Sequential` model, fetches MobileNetV2 directly from TF-Hub, extracts the `(1792, 120)` kernel and `(120,)` bias datasets directly from `model.weights.h5`, and sets them via `model.layers[2].set_weights([kernel, bias])`.

### 2. Streamlit Cloud Starlette Compatibility (`requirements.txt`)
- **Problem**: `uv` automatically installed Starlette 1.4.0+, which introduced breaking changes to `GZipResponder.__init__()` expected by Streamlit's ASGI middleware, causing 500 server errors on health checks.
- **Solution**: Pinned `starlette<1.4.0` in `requirements.txt`.

### 3. Missing `pkg_resources` Fallback (`app.py` & `model_loader.py`)
- **Problem**: In `setuptools>=70.0.0`, `pkg_resources` was removed, causing `tensorflow_hub` to crash on import.
- **Solution**: Injected a top-level `sys.modules["pkg_resources"]` fallback shim prior to third-party imports.

### 4. EXIF Orientation Transposing (`app.py`)
- **Problem**: Mobile phone uploads contained EXIF rotation metadata (e.g., 90° sideways orientation), causing rotated features to be passed into MobileNetV2.
- **Solution**: Added `PIL.ImageOps.exif_transpose()` to auto-rotate all uploaded images upright before model preprocessing.

### 5. Streamlit Cloud Secrets Management (`gemini_service.py`)
- **Solution**: Updated `get_gemini_api_key()` to support both local `.env` files and `st.secrets["GEMINI_API_KEY"]` for cloud deployments.

---

## 📂 Repository File Structure

```
DogBreedAI/
├── app.py                      # Main Streamlit Dashboard Application & Layout
├── model_loader.py             # Robust HDF5 weight extractor & model loader
├── prediction.py               # Inference pipeline & top-K confidence ranking
├── preprocessing.py            # Image conversion, resizing (224x224), & normalization
├── gradcam.py                  # Grad-CAM heatmap generation & image blending
├── gemini_service.py           # Google Gemini AI SDK integration & chat handler
├── prompt_templates.py         # System prompt templates for Gemini AI
├── utils.py                    # Path definitions, breed title formatting, file size formatters
├── main.py                     # Model training & Kaggle pipeline script
├── evaluation_metrics.py       # Classification evaluation metrics script
├── requirements.txt            # Python package dependencies & version pins
├── requirement.txt             # Backup requirements manifest
├── runtime.txt                 # Python runtime version definition (python-3.11)
├── .env                        # Local environment secrets file (git-ignored)
├── .gitignore                  # Git exclusion rules
├── dog_breed_model.keras       # Trained model archive file
├── dog_breeds.npy              # 120 Dog breed class names array
└── README.md                   # Repository documentation
```

---

## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shlok13-cell/DogBreedAI-2.O.git
   cd DogBreedAI-2.O
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Gemini API Key**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

5. **Launch the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Live Production Deployment

- **Live Web App**: [https://dogbreedai-2o-excftjkuuxjumbme5u6pzb.streamlit.app/](https://dogbreedai-2o-excftjkuuxjumbme5u6pzb.streamlit.app/)
- **GitHub Repository**: [https://github.com/shlok13-cell/DogBreedAI-2.O](https://github.com/shlok13-cell/DogBreedAI-2.O)

---

## 🎯 Conclusion & Future Scope

The **Dog Breed Classifier AI Dashboard** successfully bridges Deep Computer Vision with Explainable AI and Generative LLMs in a single web application. Future enhancements include:
- Expanding dataset coverage to multi-dog detection in single images via YOLO/SSD object detection.
- Fine-tuning MobileNetV2 backbone weights for higher top-1 accuracy on subtle breed variations.
- Adding audio-based bark identification alongside visual classification.
