import pandas as pd
import re
import os
os.getcwd()

data = pd.read_csv('data/raw/MovieSummaries/plot_summaries.txt', delimiter='\t', header=None)
column_names = ['Wikipedia ID', 'Plot']
data.columns = column_names

# CLEANING TEXT IN THE 'PLOT' COLUMN

#1 define a regular expression pattern to match words inside {{ }}
pattern = re.compile(r'\{\{.*?\}\}', re.IGNORECASE)
# clean the 'Plot' column using the regular expression pattern
data['Plot'] = data['Plot'].apply(lambda x: re.sub(pattern, '', x))

#2 Define a regular expression pattern to match everything between <ref and }} or ]
pattern_ref = r'<ref.*?(?:\}\}|\]|\|)'
# Apply the regular expression pattern to the text in the 'Plot' column
data['Plot'] = data['Plot'].str.replace(pattern_ref, '', regex=True)

#3 Remove the pattern for links like http: * html
pattern_http = r'http\S*?html'
data['Plot'] = data['Plot'].str.replace(pattern_http, '', flags=re.IGNORECASE)

#4 replace '([[' and  '-->' symbols with an empty string in the 'Plot' column
data['Plot'] = data['Plot'].str.replace(r'\(\[\[|-->|\[\[', '', regex=True)

#5 remove the "{{plot|datePBS-AE>" symbol from the "Plot" column
data['Plot'] = data['Plot'].str.replace('{{plot\|datePBS-AE>', '')

#6 remove words between { } in the 'Plot' column
data['Plot'] = data['Plot'].apply(lambda x: re.sub(r'\{.*?\}', '', x))

#7 Replace "{{Expand section|date" with an empty string in all rows of column_name
data["Plot"] = data["Plot"].str.replace('{{Expand section|date', "")

#8 remove the string from all rows of 'Plot' column
data['Plot'] = data['Plot'].str.replace('{{cite news|url', '')

#9 remove {{ from all rows in the Plot column
data['Plot'] = data['Plot'].str.replace('{{', '')
data['Plot'] = data['Plot'].str.replace('{| class""col"" width""col"" width""col"" widthAct two - TrainingAct three - The mission', '')

# Find the rows without plot descriptions
no_plot = data[data['Plot'].isna() | (data['Plot'] == '')]

# Print the rows without plot descriptions
print(no_plot[['Wikipedia ID', 'Plot']])

# Remove the rows without plot descriptions
data = data.drop(no_plot.index)

# Read the 'Movies_metadata.csv' file into a dataframe
movies_df = pd.read_csv('data/processed/Movies_metadata.csv')

# Merge the two dataframes on the 'Wikipedia ID' column
plot_df = pd.merge(data, movies_df[['Wikipedia ID', 'Movie name']], on='Wikipedia ID', how='left')

# Remove the duplicates
plot_df = plot_df.drop_duplicates(subset='Wikipedia ID', keep='first')

# Reorder the columns and reset the indices
plot_df = plot_df[['Wikipedia ID', 'Movie name', 'Plot']]
plot_df = plot_df.dropna()
plot_df = plot_df.reset_index(drop=True)

plot_df.to_csv('Plot_descriptions.csv', index = False)