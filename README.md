# IdiomSense: Multimodal Classification of Figurative Language

This repository contains a complete implementation of a **multimodal idiom understanding and classification system**. It explores how idioms can be interpreted using both visual and textual modalities and implements a robust NLP + Vision-based solution.

---

## Objective

To build a system capable of identifying whether a sentence uses a **literal** or **idiomatic** expression, using both:
- The **sentence text**
- The **corresponding image caption**

---

## Dataset

Located under `NLP Dataset`, with the following structure:

```
NLP Dataset
├── train
│   ├── train.csv
│   └── images
├── val
│   ├── val.csv
│   └── images
├── test
│   ├── test.csv
│   └── images
```

Each record includes:
- `sentence`: Sentence to analyze
- `compound`: Idiomatic phrase
- `image_caption`: Caption of related image
- `label`: Binary value indicating idiomatic (1) or literal (0)

---

## Workflow Summary

### Step 0: Install & Setup
- CLIP installed from GitHub
- All necessary libraries loaded (PyTorch, Transformers, Seaborn, Matplotlib, etc.)
- Data loaded from Google Drive

---

### Step 1: Exploratory Data Analysis (EDA)
- Sentence and caption length distributions visualized
- POS tagging, dependency parsing, and NER using **spaCy**
- Idiom highlighting and word clouds
- Image dimension analysis
- Compound frequency visualizations

---

### Step 2: LLM-based Data Augmentation
- **GPT-2 Large** used to paraphrase idiomatic sentences
- Augmented paraphrases added to training set
- CSV with augmented rows saved for reproducibility

---

### Step 3: Zero-Shot Classification (CLIP)
- CLIP model (ViT-B/32) used to embed both sentence and image caption
- Cosine similarity between embeddings used for classification
- F1 score computed across multiple thresholds
- **Best F1 ≈ 0.72** (threshold-optimized)

---

### Step 4: Fine-tuned Classifier (Enhanced)
- **SentenceTransformer (`all-mpnet-base-v2`)** used to embed text
- Additional cosine similarity feature computed
- Custom PyTorch classifier:
  - Input: 1536-dim embeddings + similarity → 1537 total
  - Layers: Linear → LayerNorm → GELU → Dropout (x3)
  - Output: 2 classes (idiomatic/literal)
- Layer freezing/unfreezing and dynamic learning rate scheduling used
- Best model saved as `NLP_pred_model.pt`

---

## Evaluation Summary

- Model: `EnhancedClassifier`
- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Scheduler: `ReduceLROnPlateau`
- Best F1 (validation): **> 0.80**
- Early stopping enabled

---

## Visualizations & Analysis

- Violin & Box plots for sentence/caption lengths
- POS tag distribution around idioms
- WordClouds of idioms & sentence corpora
- Scatter plots for image shapes
- Confusion matrix and classification report via sklearn

---

## Technologies Used

- PyTorch  
- HuggingFace Transformers  
- SentenceTransformers  
- CLIP (OpenAI)  
- spaCy  
- Matplotlib, Seaborn, WordCloud  
- Google Colab, tqdm


---

## Reproducibility

To run this in Colab:
1. Mount Google Drive
2. Set path 
3. Run the notebook cell-by-cell

---

## Author

- **Pranav Sunil Raja**  
- Newcastle University  
- GitHub: [@pranavsraja](https://github.com/pranavsraja)

---

## License

This project is part of academic coursework. Dataset is confidential and for internal use only.
