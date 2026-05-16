# imdb-NLP

A Machine Learning project focused on Natural Language Processing (NLP) to classify the sentiment of IMDb movie reviews into positive or negative categories. This project achieves an accuracy of ~88% using traditional ML techniques.

## 📌 Project Overview
The main goal of this project is to build an end-to-end NLP pipeline. It covers everything from raw text data loading, custom regular expression cleaning, feature extraction using TF-IDF, to model training and evaluation using Logistic Regression.

## 📊 Dataset Structure
The project uses the **Large Movie Review Dataset v1.0** from IMDb, which contains 50,000 highly polar reviews split evenly into:
* **25,000 reviews for training** (12.5k positive / 12.5k negative)
* **25,000 reviews for testing** (12.5k positive / 12.5k negative)

*Note: Following the dataset documentation guidelines, training and test sets contain a completely disjoint set of movies to prevent the model from memorizing movie-unique terms (avoiding Data Leakage).*

## 🛠️ Tech Stack & Libraries
* **Python 3**
* **Pandas**: Data manipulation and DataFrame structuring.
* **Scikit-Learn**: Feature extraction (`TfidfVectorizer`), Model training (`LogisticRegression`), and evaluation metrics.
* **Regex (`re`)**: Specialized text engineering and cleaning.
* **TQDM**: Visual progress bars for ETL processing.

## 🚀 NLP Pipeline & Implementation

### 1. Data Cleaning & Preprocessing
Raw IMDb reviews contain persistent malformed HTML tags (like `<br />`) and noise. A custom preprocessing function was engineered to:
* Remove specific HTML line breaks and remaining tags using Regular Expressions.
* Strip punctuation and numbers, keeping only alphabetic characters.
* Normalize text to lowercase and remove extra whitespaces.

### 2. Feature Engineering (Vectorization)
Computers process numbers, not words. To transform the clean text into numerical matrices, I implemented **TF-IDF (Term Frequency-Inverse Document Frequency)**, limiting the vocabulary to the top 5,000 most important features. 

### 3. Model Training
A **Logistic Regression** model was trained on the vectorized training set. It serves as an incredibly fast and highly effective baseline for binary text classification.

## 📈 Results & Evaluation
By respecting the strict division between training and testing sets (ensuring no data leakage), the model achieved the following performance on completely unseen movies:

* **Official Accuracy:** ~88%

### Classification Report:
```text
              precision    recall  f1-score   support
    Negative       0.89      0.87      0.88     12500
    Positive       0.87      0.89      0.88     12500
