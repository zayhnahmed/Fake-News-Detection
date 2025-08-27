# 📰 Fake News Detector  

A **machine learning–powered news classifier** that detects whether a given news article is **Fake (0)** or **True (1)**.  
It uses **TF-IDF vectorization** and a **Random Forest Classifier (with calibration)** for improved probability estimates.  
The system also supports **incremental learning with user feedback**, retraining itself when corrections are provided.  

---

## 🚀 Features  

- **Data Preprocessing**  
  - Cleans text (removes URLs, HTML, punctuation, extra spaces)  
  - Combines news `title` + `text` for better context  

- **Model Training**  
  - Uses **TF-IDF (unigrams + bigrams)** for feature extraction  
  - **Random Forest Classifier** with calibration for probability scores  
  - Handles class imbalance with weighted training  

- **Prediction**  
  - Classifies an article as **FAKE** or **TRUE**  
  - Provides a **credibility score (%)**  

- **User Feedback & Retraining**  
  - Add misclassified articles for retraining  
  - Feedback has higher weight to improve future predictions  
  - Tracks effect of retraining on last prediction  

- **Interactive CLI**  
  - Test an article by pasting text  
  - Optionally provide feedback to improve the model  

---

## 🛠️ Tech Stack  

- **Python 3.8+**  
- **Libraries:**  
  - `numpy`, `pandas`, `scikit-learn`  
  - `re` (regex for text cleaning)  

---

## 📂 Dataset  

The project uses two CSV files:  

- **`Fake.csv`** → Collection of false news articles  
- **`True.csv`** → Collection of verified true news articles  

Each file contains:  
- `title` → Headline of the news  
- `text` → Body of the article  
- `subject` → Category/topic of the article  
- `date` → Publication date  

---

## 📦 Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```
   
2. Install dependencies:

`pip install numpy pandas scikit-learn`

3. Make sure you have the dataset files:

`Fake.csv`
`True.csv`

4. Run the program:

`python fake_news_detector.py`

▶️ Usage

1. When the program starts, it loads and trains the model on `Fake.csv` + `True.csv`.

2. Choose an option:

- 1 → Test an article
- 2 → Exit

3. Paste your article text.

4. The program outputs:

- Prediction (FAKE or TRUE)
- Credibility score (confidence %)

5. Optionally provide feedback (0 for fake, 1 for true). The model retrains with higher weight for your input.

## 📊 Example
```bash
Prediction: TRUE
Credibility Score: 92.45%

Would you like to provide feedback for training? (y/n): y
Was this article real (1) or fake (0)? 1
Model has been retrained with the new article.
```

## 🔮 Future Enhancements

- Deploy as a web app with Flask/Django + REST API
- Add deep learning models (LSTMs, Transformers)
- Implement a real-time news checker via RSS feeds
- Support for multi-language detection

📜 License

This project is licensed under the MIT License.
