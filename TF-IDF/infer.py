import pandas as pd
from sklearn.model_selection import train_test_split
from train import preprocess_data
from train import train_tf_idf

if __name__ == "__main__":
    import pandas as pd
    from io import StringIO

    data = """
    3124462,442590,66090,4
    3124463,442590,186243,3
    3124464,442590,81662,2
    3124465,442590,184624,4
    3124466,442590,63258,3
    3124467,442590,165016,3
    """

    # Đọc dữ liệu từ chuỗi CSV
    df = pd.read_csv(StringIO(data), header=None)

    # Đặt tên cột nếu cần
    df.columns = ['No_care', 'user_id', 'book_id', 'rating']

    dfbooks = pd.read_csv('/Users/phamkien/Documents/MLprojjjjj/Book-Recommendation/Data/final_books.csv')
    dfratings = pd.read_csv('/Users/phamkien/Documents/MLprojjjjj/Book-Recommendation/Data/interaction.csv')
    finalbooks, finalratings, userid_dict = preprocess_data(dfbooks, dfratings)
    
    train_df, test_df = train_test_split(finalratings,
                            stratify=finalratings['newuser_id'], 
                            test_size=0.1875,
                            random_state=42)
    model = train_tf_idf(finalbooks, train_df, userid_dict)
    print(model.predict_new_user(df))


    
