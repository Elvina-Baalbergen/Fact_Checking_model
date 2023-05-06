import pandas as pd
import os

all_movies = pd.DataFrame()

directory = "./Fact_Checking_model/data/raw/2_reviews_per_movie_raw"

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Load the movie data from the csv file
        f1 = pd.read_csv(os.path.join(directory, filename), delimiter=',')
        
        # Select only the first 10 reviews for this movie
        f1 = f1.head(10)

        # Join title and review together, remove other columns
        f1 = f1.drop(['username','rating','helpful','total','date'], axis=1)
        f1['title'] = f1['title'].apply(lambda x: x.rstrip('\n'))
        f1 ['reviews'] = f1.apply(lambda row: row['title'] + ":" + row['review'], axis = 1)
        f1 = f1.drop(['title','review'], axis=1)

        # Extract the movie name and year from the filename
        film_title = filename[:filename.find('.csv') - 5]
        film_year = filename[-8:-4]

        # Add the movie name and year as columns to the dataframe
        f1.insert(0, "title", film_title)
        f1.insert(1, "year", film_year)
        
        # Concatenate the dataframe for this movie with the dataframe for all previous movies
        all_movies = pd.concat([all_movies, f1], ignore_index=True)
    
all_movies.to_csv('./Fact_Checking_model/data/processed/reviews.csv', index=False)