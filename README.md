# Sentiment-Analysis-on-Twitter-Data-Sentiment140-
A comprehensive comparison of deep learning models for binary sentiment classification on Twitter data using the Sentiment140 dataset.
Sentiment Analysis on Twitter Data (Sentiment140)

A comprehensive comparison of different neural network architectures and transformer models for binary sentiment classification on the Sentiment140.

📁 Dataset
Sentiment140 from Kaggle:
🔗 https://www.kaggle.com/datasets/kazanova/sentiment140

📋 Project Overview
This project implements and compares multiple deep learning approaches for sentiment analysis on Twitter data:

Traditional Neural Networks: ANN, BiLSTM, CNN1D

Transformer Models: DistilBERT, RoBERTa

The models are trained on the Sentiment140 dataset containing 1.6 million tweets labeled as positive or negative sentiment.

🚀 Features
Data Preprocessing: Comprehensive text cleaning and normalization

Multiple Architectures:

ANN with Global Average Pooling

Bidirectional LSTM (BiLSTM)

1D Convolutional Neural Network (CNN1D)

DistilBERT Transformer

RoBERTa Transformer

Model Comparison: Detailed performance metrics and visualization

Reproducible: Fixed random seeds and standardized evaluation

📊 Dataset
Sentiment140: 1.6 million tweets with binary labels:

0 → Negative sentiment

4 → Positive sentiment

Data Preprocessing
URL removal

Mention removal (@user)

Hashtag normalization

Special character cleaning

Text normalization

🏗️ Model Architectures
1. ANN (Artificial Neural Network)
Embedding layer (128 dimensions)

Global Average Pooling

Dense layers with dropout

2. BiLSTM (Bidirectional LSTM)
Embedding layer

Bidirectional LSTM (64 units)

Dense classification layer

3. CNN1D
Embedding layer

1D Convolutional layer (128 filters)

Global Max Pooling

Dense layers

4. DistilBERT
Pretrained DistilBERT-base-uncased

Fine-tuned on sentiment data

Sequence classification head

5. RoBERTa
Pretrained RoBERTa-base

Fine-tuned for sentiment analysis

Advanced transformer architecture

📈 Performance Metrics
All models are evaluated on:

Accuracy

F1-score (weighted)

Precision (weighted)

Recall (weighted)

🛠️ Installation & Requirements
bash
# Install dependencies
pip install tensorflow==2.19.0 tensorflow-text==2.19.0 transformers==4.57.1
pip install pandas numpy seaborn matplotlib scikit-learn wordcloud evaluate
pip install datasets torch
💻 Usage
Data Preparation:

python
# Load and preprocess Sentiment140 dataset
df = pd.read_csv("training.1600000.processed.noemoticon.csv")
Train Models:

python
# Example: Train ANN model
ann = build_ann()
history_ann = ann.fit(X_train, y_train, validation_data=(X_val, y_val))
Evaluate:

python
# Get predictions and metrics
preds = model.predict(X_test)
print(classification_report(y_test, preds))
Compare Models:

python
# Generate comparison table
df_comparison = pd.DataFrame(metrics_list)
📊 Expected Results
The transformer models (DistilBERT and RoBERTa) typically achieve the highest performance, followed by BiLSTM and CNN architectures. The comparison includes:

Accuracy scores across all models

Confusion matrices

Training/validation curves

Word frequency analysis

🎯 Key Findings
Transformers generally outperform traditional architectures

BiLSTM shows strong performance for sequential text data

CNN1D provides good balance of speed and accuracy

ANN serves as a strong baseline model

📁 Project Structure
text
sentiment-analysis/
├── sentiment140_analysis.ipynb  # Main notebook
├── requirements.txt             # Dependencies
├── models/                      # Saved models
├── logs/                       # Training logs
└── results/                    # Evaluation results
🔧 Configuration
Max Sequence Length: 80 tokens

Vocabulary Size: 30,000

Embedding Dimension: 128

Batch Size: 256 (traditional), 16 (transformers)

Training Epochs: 2-4

📝 Notes
The project uses stratified sampling to maintain class balance

Early stopping prevents overfitting

All models use the same preprocessing pipeline

Results are reproducible with fixed random seeds

🤝 Contributing
Feel free to contribute by:

Adding new model architectures

Improving preprocessing pipeline

Enhancing visualization capabilities

Optimizing hyperparameters


🙏 Acknowledgments
Sentiment140 dataset creators

Hugging Face for transformer models

TensorFlow and PyTorch teams
