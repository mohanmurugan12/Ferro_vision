# Ferro Vision: Remaining Useful Life (RUL) Forecasting & Visual Defect Detection

An end-to-end deep learning suite built for industrial asset reliability, predictive maintenance, and automated manufacturing quality inspection.

## 📌 Project Overview
Industrial asset reliability combines prognostics (predicting failure before it occurs) with visual quality inspection. Ferro Vision addresses three main pillars:
1. **Turbofan RUL Regression (ANN):** Predicts remaining operating cycles using static per-cycle sensor snapshots.
2. **Turbofan Sequence Regression (LSTM):** Evaluates temporal sensor degradation using sliding window sequences.
3. **Casting Defect Classification (CNN):** Inspects submersible pump impeller casting images to detect manufacturing flaws (Normal vs. Defective).

---

## 🚀 Key Features & Architecture
- **Frameworks Used:** Python, PyTorch, Streamlit, Pandas, NumPy.
- **Models Implemented:** 
  - Artificial Neural Network (ANN) for tabular regression.
  - Long Short-Term Memory (LSTM) network for sequence-to-sequence modeling.
  - Convolutional Neural Network (CNN) for binary image classification.
- **Interactive Web App:** A multi-page Streamlit application (`app.py`) providing EDA insights, RUL predictions, and real-time casting quality inspection.

---

## 🛠️ Project Structure
```text
Ferro_Vision/
│
├── data/                               # Turbofan and Casting datasets
├── notebooks/                          # Jupyter notebooks for cleaning, EDA, and training
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_rul_regression_ann.ipynb
│   ├── 03_rul_seq2seq_rnn_lstm.ipynb
│   └── 04_casting_defect_cnn.ipynb
│
├── saved_models/                       # Trained PyTorch state dicts (.pth)
│   ├── rul_lstm_model.pth
│   └── casting_cnn_model.pth
│
├── app.py                              # Main Streamlit Multi-page Application
├── requirements.txt                    # Python dependencies
└── README.md                           # Project Documentation


pip install -r requirements.txt

streamlit run app.py