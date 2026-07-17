# KrishiNetra-AI 🌿: Modular Multi-Crop Plant Disease Diagnostics Framework

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-v3-red?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

An enterprise-grade Deep Learning system engineered for automated, multi-crop plant disease identification. Moving away from a single bloated network, **KrishiNetra-AI** deploys a **modular architecture of dedicated, crop-specific CNN models**. Powered by an interactive Streamlit engine, it delivers high-confidence, isolated health analytics across five major agricultural crop domains.

---

## 1. Abstract

Modern precision agriculture demands fast, accurate, and scalable disease identification frameworks to mitigate crop loss. Monolithic multi-class models often suffer from inter-class feature confusion when trained simultaneously on botanically diverse crops (e.g., confusing Apple fungal lesions with Tomato bacterial spots).

**KrishiNetra-AI** solves this fundamental bottleneck by introducing a **modular multi-model execution pipeline**. Developed through a structured two-developer collaborative workflow, the system segregates diagnostics into **five independent crop domains — Apple, Grapes, Tomato, Corn, and Potato**. Each crop features its own standalone neural network optimized specifically for its distinct botanical pathologies. By eliminating cross-crop feature contamination and employing a gradual feature-reduction layer, the individual models retain high fine-grained feature sensitivity, resulting in exceptionally stable, production-ready inference — with every one of the five models achieving **90%+ test accuracy**, and four of the five crossing **98%**.

The link to access the complete dataset :

---

## 2. Why We Pivoted: From a Single Unified Classifier to Modular Per-Crop Models

The project didn't start out modular. Our original plan was simpler on paper: **one single classifier trained across all 19 disease/healthy classes spanning all five crops**, so the whole system could be shipped as a single model file and a single inference call.

In practice, this unified approach **did not perform well**. Training one network across botanically unrelated crops meant the model had to learn wildly different leaf textures, lesion shapes, and background noise patterns simultaneously — and disease symptoms that are visually similar across crops (e.g., blight-like lesions on Corn vs. Potato vs. Tomato) started bleeding into each other's decision boundaries. Accuracy and generalization suffered as a result.

Instead of pushing a bigger, deeper monolithic network to compensate, we **re-architected the system around crop-specific isolation**:

* Each crop got its **own dedicated CNN**, trained only on its own disease classes — removing all cross-crop feature contamination.
* Each model could be independently tuned, debugged, and re-trained without affecting the other four.
* The Streamlit front-end was rebuilt to **dynamically route** a user's uploaded image to the correct crop-specific model, instead of relying on one global decision boundary.

This pivot is the reason every model in this repo — Apple, Grapes, Tomato, Corn, and Potato — now hits **90%+ test accuracy independently**, something the original single-classifier approach could not achieve.

---

## 3. Aim and Objectives

### Primary Aim
To construct a modular, production-ready computer vision pipeline that hosts specialized deep learning models for individual crops — preventing representational cross-contamination and optimizing inference precision across a broad botanical scope.

### Core Technical Objectives
* **Objective 1 — Multi-Crop Classification:** Deliver disease diagnostics across **5 distinct plant domains** (Apple, Grapes, Tomato, Corn, Potato), using independent, specialized training pipelines per crop instead of a single monolithic multi-class setup.
* **Objective 2 — Lightweight Custom CNN Backbones:** Implement custom, lightweight convolutional backbones per crop to minimize parameter overhead while retaining strong spatial feature extraction, using a Global Average Pooling (GAP) layer to enforce spatial invariance.
* **Objective 3 — Information Bottleneck Resolution:** Structure the dense classification head of each standalone network into a **Gradual Funnel Layout** ($512 \rightarrow 256 \rightarrow 64 \rightarrow \text{Output}$), ensuring smooth mathematical transitions into the final class space instead of an abrupt dimensional collapse.
* **Objective 4 — Multi-Model Production Routing:** Build a clean, cross-platform routing pipeline within Streamlit that dynamically loads the exact `.keras` model file required, based on the user's selected crop type.

---

## 4. Multi-Crop Dataset & Specialized Domain Matrix

The framework manages isolated datasets and independently trained models across five major crop domains, split across a two-developer engineering workflow.

### Model & Dataset Distribution Breakdown

| Developer | Target Crop | Model Status | Classes Managed | Target Pathologies Covered |
| :--- | :--- | :--- | :--- | :--- |
| **Sanchal Kumar** | Apple | model_apple_21.1.8`.keras` Model | 4 Classes | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Sanchal Kumar** | Grapes | model_grapes_1.0.2`.keras` Model | 4 Classes | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy |
| **Sanchal Kumar** | Tomato | model_tomato_1.0.1`.keras` Model | 4 Classes | Late Blight, Leaf Mold, Septoria Leaf Spot, Healthy |
| **Shreyash Kumar Sah** | Corn | corn_model_11`.keras` Model | 4 Classes | Blight, Common Rust, Gray Leaf Spot, Healthy |
| **Shreyash Kumar Sah** | Potato | potato_model_18`.keras` Model | 3 Classes | Early Blight, Late Blight, Healthy |

> All five models are fully trained, evaluated, and integration-complete — see the benchmark results in Section 6 below.

### Leaf Pathological Matrix Visualizer
*(Representative mosaic highlighting unique textures, chlorotic spots, and varying background noise managed independently by each dedicated model network)*

### 1.Apple 
### (a)Apple Cedar Rust
<p align="center">
  <img src="07_Assets/Apple Stat/Apple_CD (1).jpg" width=24%>
  <img src="07_Assets/Apple Stat/Apple_CD (1) - Copy.jpg" width=24%>
</p>

### (b)Apple Black Rot
<p align="center">
  <img src="07_Assets/Apple Stat/Apple_Diseased_BLK_RT (1).png" width=24%>
  <img src="07_Assets/Apple Stat/Apple_Diseased_BLK_RT (1) - Copy.png" width=24%>
  <img src="07_Assets/Apple Stat/Apple_Diseased_BLK_RT (3).png" width=24%>
  <img src="07_Assets/Apple Stat/Apple_Diseased_BLK_RT (3) - Copy.png" width=24%>
</p>

### (c)Apple Scab
<p align="center">
  <img src="07_Assets/Apple Stat/Apple_scb (2).png" width=24%>
  <img src="07_Assets/Apple Stat/Apple_scb (2) - Copy.png" width=24%>
</p>

### (c)Apple Healthy
<p align="center">
  <img src="07_Assets/Apple Stat/Apple_HL_3.jpg" width=24%>
  <img src="07_Assets/Apple Stat/Apple_HL_3 - Copy.png" width=24%>
</p>

### 2.Grapes
### (a)Grape Esca Black Measles
<p align="center">
  <img src="07_Assets/Grape Stats/Black Measles (3).png" width=24%>
  <img src="07_Assets/Grape Stats/Black Measles (3) - Copy.png" width=24%>
</p>

### (b)Grape Black Rot
<p align="center">
  <img src="07_Assets/Grape Stats/Grape_BLACK R @ (4).jpg" width=24%>
  <img src="07_Assets/Grape Stats/Grape_BLACK R @ (4) - Copy.jpg" width=24%>
</p>

### (c)Grape Leaf blight(Isariopsis_Leaf_Spot)
<p align="center">
  <img src="07_Assets/Grape Stats/Grape_Leaf_spot.jpg" width=24%>
  <img src="07_Assets/Grape Stats/Grape_Leaf_spot - Copy.jpg" width=24%>
</p>

### (d)Grape Healthy
<p align="center">
  <img src="07_Assets/Grape Stats/Grape_H_L (1).jpeg" width=24%>
  <img src="07_Assets/Grape Stats/Grape_H_L (1) - Copy.jpeg" width=24%>
</p>

### 3.Tomato
### (a)Tomato Septoria leaf spot
<p align="center">
  <img src="07_Assets/Tomato Stats/Tom_sep_spot (2).png" width=24%>
  <img src="07_Assets/Tomato Stats/Tom_sep_spot (2) - Copy.png" width=24%>
</p>

### (b)Tomato Late blight
<p align="center">
  <img src="07_Assets/Tomato Stats/Tomato _LB2.jpg" width=24%>
  <img src="07_Assets/Tomato Stats/Tomato _LB2 - Copy.jpg" width=24%>
</p>

### (c)Tomato Leaf Mold
<p align="center">
  <img src="07_Assets/Tomato Stats/Tomato_leaf_mo.jpg" width=24%>
  <img src="07_Assets/Tomato Stats/Tomato_leaf_mo - Copy.jpg" width=24%>
</p>

### (d)Tomato Healthy
<p align="center">
  <img src="07_Assets/Tomato Stats/tomato Helat.jpg" width=24%>
  <img src="07_Assets/Tomato Stats/tomato Helat - Copy.jpg" width=24%>
</p>

### 4.Corn(maize)
### (a)Corn Common Rust
<p align="center">
  <img src="07_Assets/Corn Stats/Corn Rust12.jpg" width=24%>
  <img src="07_Assets/Corn Stats/Corn Rust12 - Copy.jpg" width=24%>
</p>

### (b)Corn Northern Leaf Blight
<p align="center">
  <img src="07_Assets/Corn Stats/Corn LB4 - Copy2.png" width=24%>
  <img src="07_Assets/Corn Stats/Corn LB4 - Copy.png" width=24%>
  <img src="07_Assets/Corn Stats/Corn LB5.jpg" width=24%>
</p>

### (c)Corn Cercospora Gray leaf spot
<p align="center">
  <img src="07_Assets/Corn Stats/Corn Leaf spot.jpg" width=24%>
  <img src="07_Assets/Corn Stats/Corn Leaf spot - Copy.jpg" width=24%>
</p>

### (d)Corn Healthy
<p align="center">
  <img src="07_Assets/Corn Stats/Corn HL2.jpg" width=24%>
  <img src="07_Assets/Corn Stats/Corn HL2 - Copy.jpg" width=24%>
</p>

### 5.Potato
### (a)Potato Early Blight
<p align="center">
  <img src="07_Assets/Potato Stats/Potato EarlyB2.jpg" width=24%>
  <img src="07_Assets/Potato Stats/Potato EarlyB2 - Copy.jpg" width=24%>
  <img src="07_Assets/Potato Stats/Potato EB5.jpg" width=24%>
</p>

### (b)Potato Late Blight
<p align="center">
  <img src="07_Assets/Potato Stats/Potato LB2 - Copy.jpg" width=24%>
  <img src="07_Assets/Potato Stats/Potato LB2.jpg" width=24%>
</p>

### (d)Potato Healthy
<p align="center">
  <img src="07_Assets/Potato Stats/Potato_HL243.jpg" width=24%>
  <img src="07_Assets/Potato Stats/Potato_HL243 - Copy.jpg" width=24%>
</p>


### 📋 Per-Crop Class-Wise Image Distribution

The table below breaks down how many images were used per disease class, per crop. **Train / Validation counts are left blank for you to fill in** — the Test column is already populated from each model's classification report (`support` values), which is the real number of test-set images per class.

**Apple** (Total test set: 740)
| Class | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| Apple___Apple_scab | 1306 | 373 | 186 | 1865 |
| Apple___Black_rot | 1337 | 381 | 190 | 1908 |
| Apple___Cedar_apple_rust | 1237 | 353 | 176 | 1766 |
| Apple___healthy | 1320 | 376 | 188 | 1884 |
| **Total** | | | **7423** | |

**Grapes** (Total test set: 1070)
| Class | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| Grape___Black_rot |1237 | 263 | 263 | 1763 |
| Grape___Esca_(Black_Measles) | 1296 | 277 | 277 | 1850 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1294 | 276 | 276 | 1846 |
| Grape___healthy | 1188 | 254 | 254 | 1696 |
| **Total** | | | **7155** | |

**Tomato** (Total test set: 1235)
| Class | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| Tomato___Late_blight | 1353 | 253 | 327 | 1933 |
| Tomato___Leaf_Mold | 1293 | 244 | 282 | 1819 |
| Tomato___Septoria_leaf_spot | 1309 | 247 | 319 | 1875 |
| Tomato___healthy | 1384 | 259 | 307 | 1950 |
| **Total** | | | **7577** | |

**Potato** (Total test set: 161)
| Class | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| Potato___Early_blight | 838 | 157 | 53 | 1048 |
| Potato___Late_blight | 900 | 168 | 58 | 1126 |
| Potato___healthy | 800 | 150 | 50 | 1000 |
| **Total** | | | **3174** | |

**Corn** (Total test set: 236)
| Class | Train | Validation | Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| Corn___Blight | 984 | 184 | 63 | 1231 |
| Corn___Common_Rust | 1095 | 205 | 69 | 1369 |
| Corn___Gray_Leaf_Spot | 693 | 130 | 44 | 867 |
| Corn___Healthy | 968 | 181 | 61 | 1210 |
| **Total** | | | **4677** | |

---

## 5. Custom Production Inference Pipeline

The framework uses an isolated prediction pipeline that allows developers to run variable batch checks using cross-platform safe path configurations, routing inputs directly to the crop's specific model graph.

```python
# Production Custom Batch Evaluator Pipeline Syntax
# Independent model routing via clean forward-slash cross-platform path handling
my_test_images = [
    "C:/Users/.../Test_Images_Real_World/Apple Testing/Apple_CD (3).jpg",
    "C:/Users/.../Test_Images_Real_World/Apple Testing/Apple_H_L.jpg",
    "C:/Users/.../Test_Images_Real_World/Apple Testing/Apple_scb (2).png"
]

# Trigger the grid visualizer execution block for the dedicated Apple model
predict_and_plot_batch(my_test_images)
```

### 🔬 Dynamic Validation Output Grid
The specialized `predict_and_plot_batch` function normalizes the input matrix, maps it onto the crop's isolated network graph, and plots class confidence profiles dynamically via an automated Matplotlib matrix plot.

### Apple
<p align="center">
  <img src="07_Assets/Apple Stat/apple.png">
</p>

### Grapes
<p align="center">
  <img src="07_Assets/Grape Stats/grapes.png">
</p>

### Tomato
<p align="center">
  <img src="07_Assets/Tomato Stats/tomato.png">
</p>

### Corn(Maize)
<p align="center">
  <img src="07_Assets/Corn Stats/corn.png">
</p>

### Potato
<p align="center">
  <img src="07_Assets/Potato Stats/potato.png">
</p>

---

## 6. Model Benchmark Results (Per-Crop)

Every model was evaluated independently on a held-out test set. Below is the consolidated benchmark across all five crop-specific networks — the direct payoff of the modular pivot described in Section 2.

### Consolidated Test Performance

| Crop | Test Accuracy | Test Loss | Overall (Weighted) Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Apple | **99.19%** | 0.0307 | 0.99 | 0.99 | 0.99 |
| Grapes | **99.44%** | 0.0148 | 0.99 | 0.99 | 0.99 |
| Tomato | **98.38%** | 0.0398 | 0.98 | 0.98 | 0.98 |
| Potato | **98.76%** | 0.0205 | 0.99 | 0.99 | 0.99 |
| Corn | **90.68%** | 0.2239 | 0.91 | 0.91 | 0.91 |

> **Note on Corn:** The Corn model trails the rest of the fleet in accuracy, largely driven by feature overlap between the `Gray_Leaf_Spot` and `Blight` classes (visible in its confusion matrix). This is a known limitation flagged for future fine-tuning — e.g. targeted augmentation or a deeper backbone for that class pair.


### Apple Model
<p align="center">
  <img src="07_Assets/Apple Stat/_Confusion_Matrix.png""
</p>
<p align="center">
  <img src="07_Assets/Apple Stat/Classification_Report.png">
</p>

### Grapes Model
<p align="center">
  <img src="07_Assets/Grape Stats/Confusion Matrix.png">
</p>
<p align="center">
  <img src="07_Assets/Grape Stats/Precision- Recall.png">
</p>

### Tomato Model
<p align="center">
  <img src="07_Assets/Tomato Stats/Confusion Matrix.png">
</p>
<p align="center">
  <img src="07_Assets/Tomato Stats/Precision - Recall.png">
</p>

### Potato Model
<p align="center">
  <img src="07_Assets/Potato Stats/potato_CM.png">
</p>
<p align="center">
  <img src="07_Assets/Potato Stats/Potato_CR.png">
</p>

### Corn Model
<p align="center">
  <img src="07_Assets/Corn Stats/corn_CM.png">
</p>
<p align="center">
  <img src="07_Assets/Corn Stats/Corn_CR.png">
</p>

---


## 7. Repository Directory Structure

```
Streamlit-KrishiNetra/
│
├── 01_app/
│   └── app1.py                          # Streamlit interface — dynamically routes and loads the correct crop model
│
├── 03_Notebooks/                        # Research & Diagnostics Hub
│   │
│   ├── 01_Experiments/                  # Early-stage model optimization & architecture trials
│   │   ├── 01_Augmentation.ipynb
│   │   ├── 02_Create_CNN_structure_01.ipynb
│   │   ├── 03_Apple_Training_01.ipynb
│   │   ├── 03_Apple_Training_02.ipynb
│   │   ├── 05_Tomato_Training_1.ipynb
│   │   ├── 05_Tomato_Training_2.ipynb
│   │   └── 05_Tomato_Training_3.ipynb
│   │
│   └── 02_Production_Final/             # Validated, production-ready per-crop notebooks
│       ├── Prediction_File_Images/      # Real-world test images used for batch inference demos
│       │   ├── Apple Predict/
│       │   ├── Corn(Maize) Predict/
│       │   ├── Grapes Predict/
│       │   ├── Potato Predict/
│       │   └── Tomato Predict/
│       ├── 00_Split_data_70_15_15.py
│       ├── 00_train_test_validate_corn.ipynb
│       ├── 00_train_test_validate_potato.ipynb
│       ├── 01_Augmentation_Enhanced.ipynb
│       ├── 02_Create_CNN_structure_01.ipynb
│       ├── 03_Apple_Training__Final.ipynb
│       ├── 03_II_Apple_Prediction.ipynb
│       ├── 04_Grapes_Training__Final.ipynb
│       ├── 04_II_Grapes_Prediction.ipynb
│       ├── 05_Tomato_Training__Final.ipynb
│       ├── 05_Tomato_vPrediction.ipynb
│       ├── 06_Corn_Training__Final.ipynb
│       ├── 06_Corn_vPrediction.ipynb
│       ├── 07_Potato_Training__Final.ipynb
│       └── 07_Potato_vPrediction.ipynb
│
├── 05_Complete_19_Class_Training/        # 🗄️ Legacy: the original unified 19-class classifier
│   ├── 01_split_data_80_20.py            # (see Section 2 — this approach was superseded by the
│   ├── 02_Training_02.py                 #  per-crop modular models above; kept for reference.
│   ├── 02_Training_03.ipynb              #  Has its own README.md documenting that experiment.)
│   ├── 02_Training_04.ipynb
│   ├── 02_Training_05.ipynb
│   ├── 03_Predict.py
│   └── README.md
│
├── 07_Assets/                                # Visual README assets (confusion matrices, graphs, dataset mosaics)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> 📁 The `05_Complete_19_Class_Training/` folder holds the **original, deprecated single-classifier experiment** referenced in Section 2 (the one trained across all 19 classes for all 5 crops). It has its own `README.md` documenting that specific experiment in detail — this top-level README focuses on the current, production per-crop architecture.

---

## 8. Installation and Local Deployment

**1. Clone the Repository**
```bash
git clone https://github.com/Shreyash71-byte/Streamlit-KrishiNetra.git
cd Streamlit-KrishiNetra
```

**2. Configure Your Isolated Python Environment**
```bash
# Create a dedicated runtime environment
python -m venv krishi_env

# Activate the local runtime (Windows)
krishi_env\Scripts\activate

# Activate the local runtime (Mac/Linux)
source krishi_env/bin/activate
```

**3. Install Required Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Production Streamlit App**
```bash
python -m streamlit run 01_app/app1.py
```

---

## 9. Engineering Team & Contribution Split

KrishiNetra-AI is an open-source collaboration designed and maintained by a two-person engineering team, with a clearly defined separation of model development roles:

**Sanchal Kumar** ([@Sanchal-01](https://github.com/Sanchal-01))
- Engineered the global production-level routing architecture and cross-platform Streamlit user interface.
- Developed, trained, and structurally optimized the standalone deep learning networks for **Apple, Grapes, and Tomato**.
- Designed and implemented the cross-platform pathing modules and the dynamic batch-inference visualizer function.

**Shreyash Kumar Sah** ([@Shreyash71-byte](https://github.com/Shreyash71-byte))
- Conducted dataset acquisition, image parsing, and advanced augmentation routines.
- Developed, trained, and validated the standalone deep learning networks for **Corn and Potato**.
- Managed training-optimization and validation benchmarking across the fleet of models, and co-led the pivot from the original unified classifier to the current modular architecture.

---

## 10. License

Copyright (c) 2026 Sanchal Kumar & Shreyash Kumar Sah

This project is shared for educational and personal learning purposes only.

You may:
- View the code
- Learn from it

You may NOT:
- Reupload this project as your own
- Use this project commercially
- Redistribute this project without permission

The software is provided "as is", without warranty of any kind.


# NOTE For permissions beyond personal learning use, please contact the author.
