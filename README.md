# Personalised_Book_Recommender

This project is a **Book Recommendation System** that suggests books based on:
- 📈 **Popularity** (Top 50 books rated by users)  
- 👥 **Collaborative Filtering** (books similar to the one you like)

It uses the **Book-Crossing Dataset** (from Kaggle) containing books, users, and ratings.

---

## Features
- Recommend **Top 50 Most Popular Books** with cover images, titles, and authors.
- Suggest **Similar Books** using **Collaborative Filtering** with **Cosine Similarity**.
- Interactive **Streamlit Web App** with sidebar navigation.
- Dataset and model files saved as `.pkl` for faster deployment.

---

## Dataset
The dataset is taken from **[Kaggle - Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)**.  
It includes:
- `Books.csv` → Book details (title, author, publication year, image URL)  
- `Users.csv` → User demographics (age, location)  
- `Ratings.csv` → User ratings for books  

---

## Tech Stack
- **Python**   
- **Pandas, Numpy** → Data preprocessing  
- **Scikit-learn** → Similarity calculation  
- **Streamlit** → Web App  
- **Pickle** → Model export  

---
