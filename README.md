# 📚 Book Recommendation System

> A machine learning–powered Streamlit application that recommends similar books using collaborative filtering and the **NearestNeighbors** algorithm.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Recommendation%20System-green)](https://scikit-learn.org/)

## 🚀 Live Demo

**[Open the Book Recommendation System](https://book-recommendation-system-6xjx.onrender.com/)**

## 🎯 Project Overview

Recommendation systems help users discover relevant content from a large collection of choices. This project builds a book recommendation application that uses historical book-rating interactions to identify books with similar user-rating patterns.

The application provides a simple web interface where a user can select a book and receive recommendations for similar titles.

## 🧠 Approach

This project focuses on **collaborative filtering**. The recommendation pipeline is based on book-rating interactions and uses **NearestNeighbors** to find books with similar patterns in the transformed user-item data.

### Recommendation workflow

```text
Book & Rating Data
        ↓
Data Loading & Cleaning
        ↓
User–Book Interaction Matrix
        ↓
Feature Transformation
        ↓
NearestNeighbors Model
        ↓
Nearest Books
        ↓
Streamlit Recommendation UI
```

### Why collaborative filtering?

Collaborative filtering can recommend items based on patterns in user-item interactions rather than relying only on book metadata. This makes it useful when users with similar preferences have interacted with similar books.

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** – data processing
- **Scikit-learn** – NearestNeighbors and machine learning
- **Streamlit** – interactive web application
- **Jupyter Notebook** – model development and experimentation

## 📊 Dataset

The project uses the **Book Recommendation Dataset** available through Kaggle.

**Dataset:** https://www.kaggle.com/ra4u12/bookrecommendation

## 📁 Project Structure

```text
Book-Recommendation-System/
├── app.py
├── Books Recommender.ipynb
├── model.pkl
├── requirements.txt
├── demo/
│   ├── Books.jpeg
│   ├── Demo_1.png
│   ├── Demo_2.png
│   └── Demo_3.png
└── README.md
```

## 🖥️ Application Demo

### Home / Recommendation Interface

![Book Recommendation System](demo/Demo_1.png)

### Recommendation Results

![Recommendation Results](demo/Demo_2.png)

![Recommendation Results](demo/Demo_3.png)

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kattubadimohammad/Book-Recommendation-System.git
cd Book-Recommendation-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔬 Model Development

The model was developed in **`Books Recommender.ipynb`**. The notebook contains the data preparation and recommendation-model workflow used to generate the serialized model artifact.

The application then loads the trained model and uses nearest-neighbor search to return similar books.

## ⚠️ Recommendation-System Considerations

Collaborative filtering has several practical limitations:

- **Cold-start problem:** new users or books may have insufficient interaction history.
- **Popularity bias:** frequently rated books can be recommended more often.
- **Sparse interactions:** large user-item matrices can contain many missing values.
- **Scalability:** nearest-neighbor search can become more expensive as the interaction dataset grows.

Potential improvements include hybrid recommendations, better similarity metrics, matrix-factorization methods, popularity-aware ranking, and offline evaluation using metrics such as **Precision@K** and **Recall@K**.

## 🔮 Future Improvements

- Add Precision@K and Recall@K evaluation.
- Introduce a content-based recommendation component.
- Build a hybrid recommendation engine.
- Improve handling of cold-start users and books.
- Add recommendation explanations.
- Add automated testing and CI.
- Improve deployment and dependency reproducibility.

## 👨‍💻 Author

**Kattubadi Mohammad**  
Data / Machine Learning Portfolio

[GitHub](https://github.com/kattubadimohammad)

---

⭐ If you find this project useful, consider starring the repository.
