# 🛡️ Ferro Vision: Industrial Asset Monitoring & Defect Detection

Ferro Vision is an end-to-end industrial intelligence platform designed for predictive maintenance of industrial equipment and automated manufacturing defect inspection using Deep Learning and Streamlit.

---

## 🚀 Key Modules

1. **Exploratory Data Analysis (EDA) & Data Cleaning**:
   - Analyzes industrial sensor time-series data and casting manufacturing image distributions.
   - Handles missing values, data normalization, and statistical summaries.

2. **Turbofan Engine RUL Forecasting (Regression / LSTM)**:
   - Utilizes an **LSTM (Long Short-Term Memory)** neural network to predict the Remaining Useful Life (RUL) of industrial turbofan engines using multi-sensor time-series logs.

3. **Casting Defect Detection & Evaluation (CNN Classification)**:
   - Employs a **Convolutional Neural Network (CNN)** classifier to inspect cast product images and distinguish between healthy (`normal`) and flawed (`defective`) components with high precision.

---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch, Torchvision
- **Web Application Interface**: Streamlit
- **Data Manipulation & Analysis**: Pandas, NumPy, Scikit-Learn
- **Image Processing**: Pillow (PIL)

---

## 📂 Project Structure

```text
Ferro Vision/
│
├── data/
│   ├── casting_data/                  # Casting product images (train/test)
│   └── Industrial Risk and Reliability Assessment/  # Turbofan sensor logs
│
├── models/
│   ├── turbofan_lstm_model.pth        # Saved LSTM weights for RUL prediction
│   └── casting_cnn_model.pth          # Saved CNN weights for defect detection
│
├── notebooks/
│   ├── 01_turbofan_cleaning_eda.ipynb # Data cleaning and EDA
│   ├── 02_turbofan_model_training.ipynb# LSTM model training & evaluation
│   └── 03_casting_defect_detection.ipynb# CNN model training & evaluation
│
├── app.py                             # Interactive 3-page Streamlit web application
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation

pip install -r requirements.txt

streamlit run app.py