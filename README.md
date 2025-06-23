# 📰 News Article Classification: Fake vs Real using BERT

## Objective
This project classifies news statements as **Fake** or **Real** using a fine-tuned BERT transformer model trained on the LIAR dataset. It includes an interactive web interface built using **Streamlit**.

---

## ⚙ Features
- Fine-tuned BERT (`bert-base-uncased`) model using Hugging Face
- Binary classification: Fake (0), Real (1)
- Trained on the LIAR dataset (political statements)
- Real-time prediction interface via Streamlit

---

## 🧠 Model Training

Training performed using the notebook `FakeNews_Training.ipynb`.  
Label mapping:
- ❌ Fake: `pants-fire`, `false`, `barely-true`
- ✅ Real: `half-true`, `mostly-true`, `true`

---

## Dataset

The project uses the LIAR dataset in three files:
- `train.tsv`
- `valid.tsv`
- `test.tsv`

These are included in the repository.

---

## 📥 Download Trained Model

Due to GitHub's 100MB file limit, the trained BERT model is hosted externally.

👉 **[Click here to download the trained model (Google Drive)]([https://drive.google.com/YOUR_MODEL_LINK](https://drive.google.com/drive/folders/1wO1vjVn8gJagZQ1P-eJcL-Mc8doxeBVe?usp=drive_link))**  
*(Replace the above link with your actual Google Drive model link.)*

### After downloading:
1. Extract `saved_fake_news_model.zip`
2. Place the extracted folder in your project directory


---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/News_Article_Classification.git
cd News_Article_Classification
```

### 2. Install Required Libraries

``` bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run streamlitapp.py

```


