import pandas as pd

from sklearn.model_selection import train_test_split

from TF_IDF.model import TfIdf

def preprocess_data(dfbooks, dfratings):
    useronly=dfratings.groupby(by= 'user_id', as_index=False).agg({'rating':pd.Series.count}).sort_values('rating',ascending = False).head(15000)
    finalratings = dfratings[dfratings.user_id.isin(useronly.user_id)]
    #chỉ lấy rating từ những người dùng đó
    bookonly = finalratings.groupby(by= 'book_id', as_index=False).agg({'rating':pd.Series.count}).sort_values('rating',ascending = False).head(8000)
    #Tìm 8,000 sách được đánh giá nhiều nhất:
    finalratings = finalratings[dfratings.book_id.isin(bookonly.book_id)]

    finalbooks = dfbooks[dfbooks.book_id.isin(bookonly.book_id)]
    #lọc finalbooks sao cho chỉ còn những sách mà nằm trong danh sách bookonly (tức là 8,000 cuốn sách bạn chọn ở bước trước
    finalbooks = finalbooks.reset_index(drop=True)
    #reset lại index từ 2, 5, 20 -> 0, 1, 2,..., drop=True để không giữ lại cột index cũ.
    finalbooks["newbookid"] = finalbooks.index + 1

    finalratings = finalratings.merge(finalbooks[['book_id','newbookid']], how='left', on= ['book_id'])
    finalratings['newuser_id'] = finalratings.groupby('user_id').grouper.group_info[0]+1
    userid_dict = {u: v for u,v in zip(finalratings['user_id'].values.tolist(), finalratings['newuser_id'].values.tolist())}

    finalbooks['text'] = finalbooks['description'].fillna(finalbooks['title'])
    finalbooks['text'] = finalbooks['text'].str.replace(r'[^\w\s]', '', regex=True).str.lower()

    return finalbooks, finalratings, userid_dict

def train_tf_idf(finalbooks, train_ratings, userid_dict):
    model = TfIdf(finalbooks=finalbooks, finalratings=train_ratings, userid_dict=userid_dict)
    return model


if __name__ == "__main__":
    dfbooks = pd.read_csv('/Users/phamkien/Documents/MLprojjjjj/Book-Recommendation/Data/final_books.csv')
    dfratings = pd.read_csv('/Users/phamkien/Documents/MLprojjjjj/Book-Recommendation/Data/interaction.csv')
    finalbooks, finalratings, userid_dict = preprocess_data(dfbooks, dfratings)
    train_df, test_df = train_test_split(finalratings,
                            stratify=finalratings['newuser_id'], 
                            test_size=0.1875,
                            random_state=42)
    model = train_tf_idf(finalbooks, train_df)

    model.eval_metrics(test_df)

