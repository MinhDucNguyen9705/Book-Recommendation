import pandas as pd 

ratings = pd.read_csv('Data/interaction.csv')
print(ratings.groupby('user_id').count().sort_values('book_id', ascending=False).head(10))