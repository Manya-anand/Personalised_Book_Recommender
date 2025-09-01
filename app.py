# import libraries
import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

st.set_page_config(layout = "wide")
st.header("Book Recommender")
st.markdown('''
### We recommend top 50 books
### We also suggest books using collaborative filtering
''')

popular = pkl.load(open('popular.pkl','rb'))
books = pkl.load(open('books.pkl','rb'))
pt = pkl.load(open('pt.pkl','rb'))
similarity_scores = pkl.load(open('similarity_scores.pkl','rb'))

# Top 50 books
st.sidebar.title("Top 50 Books")
if st.sidebar.button("SHOW"):
    cols_per_row = 5
    num_row = 10
    for row in range(num_row):
        cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            book_index = row * cols_per_row + col
            if book_index < len(popular):
                with cols[col]:
                    st.image(popular.iloc[book_index]['Image-URL-M']) # Display image of book
                    st.text(popular.iloc[book_index]['Book-Title']) # Display book title
                    st.text(popular.iloc[book_index]['Book-Author']) # Display book author

# Recommend Books
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key = lambda x : x[1], reverse=True)[1:6]
    # create empty list with book information
    #book author, book title, image url
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data

book_list = pt.index.values
st.sidebar.title("Similar Book Suggestions")
# Dropdown to select the books
selected_book = st.sidebar.selectbox("Select a book",book_list)
if st.sidebar.button("Recommend Book"):
    book_recommend = recommend(selected_book)
    cols = st.columns(5)
    for col_index in range(5):
        with cols[col_index]:
            if col_index < len(book_recommend):
                st.image(book_recommend[col_index][2])
                st.text(book_recommend[col_index][0])
                st.text(book_recommend[col_index][1])


# Show the data used
books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')  

st.sidebar.title("Data Used")
if st.sidebar.button("Show Data Used"):
    st.subheader("This is the books data used in this project.")
    st.dataframe(books)
    st.subheader("This is the user data used in this project.")
    st.dataframe(users)
    st.subheader("This is the user ratings data used in this project.")
    st.dataframe(ratings)
