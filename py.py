# import libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import data
books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

books.head()
users.head()
ratings.head()

# dimensions
books.shape
users.shape
ratings.shape

# looking for nulls
books.isnull().sum()
# droping nulls
books = books.dropna()

users.isnull().sum()
users = users.dropna()

ratings.isnull().sum()
ratings = ratings.dropna()

# checking for duplicates
books.duplicated().sum()
# drop duplicates
books = books.drop_duplicates()

users.duplicated().sum()
users = users.drop_duplicates()

ratings.duplicated().sum()
ratings = ratings.drop_duplicates()

# count unique values
books.nunique()
users.nunique()
ratings.nunique()

ratings.head()

np.sort(ratings['Book-Rating'].unique())

books.info()
books.columns
# convert year of publication to int data type
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('int32')
books.info()
users.info()
ratings.info()

# POPULARITY BASED RECOMMENDER SYSTEM
books.head()
ratings.head()
users.head()

# Joining books and ratings table
books_ratings = ratings.merge(books, on = 'ISBN')
books_ratings.head()

popular_df = books_ratings.groupby('Book-Title').agg(num_rating = ('Book-Rating','count'), avg_rating = ('Book-Rating','mean'))
popular_df = popular_df.reset_index()
popular_df

# sort based on number of ratings
popular_df.sort_values('num_rating', ascending = False)

# popularity is based on the number of people who read the book
# also based on the rating the book got
popular_df = popular_df[popular_df['num_rating']>300].sort_values('avg_rating', ascending = False)
popular_df

# top 50 books
popular_df = popular_df.head(50)
popular_df

# for the model deployment I need book-title, book-author, image-url
popular_df = popular_df.merge(books, on = 'Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_rating', 'avg_rating']]
popular_df

# COLLABORATIVE FILTERING
# similar book prediction based on user feedback
books_ratings.head()

# grouping based on user-id will tell number of books rated by each user
x = books_ratings.groupby('User-ID').count()
x
x.index
x.shape

# select only the users who have given feedback for atleast 200 books
x = x['Book-Rating']>200;
x

power_user = x[x].index
power_user
power_user.shape

# selecting only records of power users
filtered_users = books_ratings[books_ratings['User-ID'].isin(power_user)]
filtered_users

# group the best users based on book title
y = filtered_users.groupby('Book-Title').count()
y
# the above dataframe tells us how many users have read a particular book

y.sort_values('User-ID', ascending = False)

# the book rating(no. of uers) should be 50 or more 
y = y['User-ID'] >= 50
y

famous_books = y[y].index
famous_books

final_ratings = filtered_users[filtered_users['Book-Title'].isin(famous_books)]
final_ratings

# pivot table gives the rating of each book from each user
# book row with user ID as column
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt
pt = pt.fillna(0)
pt

similarity_scores = cosine_similarity(pt)
similarity_scores
type(similarity_scores)
df_temp = pd.DataFrame(similarity_scores)
df_temp
pt.index

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

print(recommend('1984'))

# EXPORT DATA AND MODEL TO PKL 
# Import pickle and dump data and model
import pickle as pkl
pkl.dump(popular_df,open('popular.pkl','wb')) # Popularity based recommender system
pkl.dump(books,open('books.pkl','wb')) # books data
pkl.dump(pt,open('pt.pkl','wb')) #feedback of books and users
pkl.dump(similarity_scores,open('similarity_scores.pkl','wb')) # similarity score of each book
