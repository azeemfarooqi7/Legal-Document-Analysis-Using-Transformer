# Legal-Document-Analysis-Using-Transformer
Enhancing Text Classification in Legal Document Analysis Using Transformer Models and Attention Mechanisms
# Legal Document Classification with Transformer Models

This project explores the use of transformer models and attention mechanisms to enhance text classification in legal document analysis. The focus is on employing models such as **BERT** and **RoBERTa**, leveraging their capabilities to handle the complexity and context-rich nature of legal texts.

## Introduction

Legal documents often present a complex structure, including long sentences, multiple references, and specific terminology. This project aims to enhance the categorization and analysis of legal texts using state-of-the-art transformer models like **BERT** and **RoBERTa**.

## Motivation

Traditional NLP methods, including rule-based systems and basic machine learning algorithms, often fail to capture the deep contextual meanings and semantic relations within legal documents. Transformer models, with their self-attention mechanisms, provide a robust solution for handling long and context-dependent sequences, making them suitable for legal text classification.

## Technologies Used

- **Programming Language:** Python
- **Frameworks:** TensorFlow, PyTorch
- **NLP Libraries:** Hugging Face’s Transformers
- **Datasets:** EURLEX57K, JRC-Acquis, ECHR Violation
- **Development Tools:** Jupyter Notebooks for experimentation and model development

## Dataset Details

The project utilizes multiple datasets tailored to legal domains:
- **EURLEX57K:** Multi-label EU legal documents dataset using the EuroVoc taxonomy.
- **JRC-Acquis:** A dataset containing a comprehensive classification system for EU legal texts.
- **ECHR Violation:** A dataset for binary classification tasks related to human rights violations.

## Model Architecture

The project implements and fine-tunes transformer models:
- **BERT** (Bidirectional Encoder Representations from Transformers) fine-tuned with domain-specific data for improved legal text classification.
- **RoBERTa** for enhanced question-answering performance on the PRIVACYQA dataset.
- Techniques such as **BiLSTM integration** for handling long documents and **label-attention mechanisms** for multi-label classification.

## Evaluation Metrics

The models are evaluated using the following metrics:
- **F1-score:** Combines precision and recall for multi-label classification.
- **Mean Reciprocal Rank (MRR):** Measures the quality of ranked results.
- **Accuracy:** Indicates the correct classification within the dataset.

## Installation

To set up the project locally:

1. Clone the repository:
git clone https://github.com/your-username/legal-text-classification.git
2. Install dependencies:
pip install -r requirements.txt
3. Run Jupyter Notebook:
jupyter notebook

## Usage

1. Preprocess the legal datasets and tokenize the text using the transformer tokenizer.
2. Fine-tune the BERT and RoBERTa models on your chosen datasets using the provided notebooks.
3. Evaluate model performance using the metrics outlined above.

## Results

- Achieved state-of-the-art F1-scores of 0.754 on the EURLEX57K dataset.
- Demonstrated a 31% improvement in answer retrieval accuracy using RoBERTa compared to traditional SVM classifiers.

## Future Work

- Explore the use of **domain-specific pretraining** on larger legal corpora.
- Implement **model distillation** for efficiency improvements and real-time application suitability.
- Investigate advanced **data augmentation techniques** for addressing data imbalance issues.

## References

- Shaheen et al. (2021) - Multi-label legal text classification using transformer models.
- Limsopatham (2021) - Adapting BERT for long legal documents.
- Vold & Conrad (2021) - Comparison of RoBERTa and traditional classifiers for legal text.
