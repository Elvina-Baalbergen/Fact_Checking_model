import pandas as pd
import os
os.getcwd()

#retrieving actors and putting them into a dataframe actors_df
actors_df = pd.read_csv('data/raw/MovieSummaries/character.metadata.tsv', delimiter='\t', header=None)
column_names = ['Wikipedia ID','Freebase movie ID','Movie release date', 'Character name', 'Actor dob', 'Gender', 'Height','Ethnicity','Actor','Age', 'Freebase character/actor map ID','Freebase character ID', 'Freebase actor ID']
actors_df.columns = column_names
actors_df.drop(['Movie release date', 'Character name', 'Actor dob', 'Gender', 'Height','Ethnicity','Age', 'Freebase character/actor map ID','Freebase character ID', 'Freebase actor ID'], axis=1, inplace=True)

#retrieving metadata (name, year, genre) and putting them into a dataframe data
data = pd.read_csv('data/raw/MovieSummaries/movie.metadata.tsv', delimiter='\t', header=None)
column_names = ['Wikipedia ID','Freebase movie ID','Movie name', 'Year', 'Movie box office revenue', 'Movie runtime', 'Movie languages', 'Movie countries', 'Genre']
data.columns = column_names
data.drop(['Freebase movie ID','Movie box office revenue', 'Movie runtime', 'Movie languages', 'Movie countries'], axis=1, inplace=True)

data['Genre'] = data['Genre'].apply(lambda x: eval(x).values())
data['Year'] = data['Year'].astype(str).str[:4]
data.loc[data['Year'] == '1010', 'Year'] = '2010'
#grouped_years = data.groupby('Year').size()
#grouped_years

#merge 2 dataframes into one df_combined
df_combined = pd.merge(actors_df, data, on = 'Wikipedia ID')
df_combined.head(9)

def select_top_actors(group):
    # sort the group by actor's count and take the top 5
    top_actors = group.sort_values(by='Actor', ascending=False).head(5)
    return top_actors

# group the dataframe by the movie column and apply the select_top_actors function
df = df_combined.groupby('Movie name').apply(select_top_actors)

# reset the index to get rid of the groupby structure
df = df.reset_index(drop=True)

new_column_order = ['Wikipedia ID',	'Freebase movie ID','Movie name', 'Year', 'Genre', 'Actor']
df = df.reindex(columns=new_column_order)

print(df.head(7))