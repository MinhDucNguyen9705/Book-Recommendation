import streamlit as st
import random
import pandas as pd
from NeuMF.inference import NeuralMatrixFactoration
from NeuMF.train import train_model
from NeuMF.utils import split_data, preprocess_data
# from TF_IDF.train import preprocess_data, train_tf_idf
import csv
import gc


# Simulated user database
if "users" not in st.session_state:
    st.session_state.users = pd.read_csv("Data/user_data.csv", index_col=0)

if "current_user" not in st.session_state:
    st.session_state.current_user = pd.DataFrame({
                                        "username": [],
                                        "password": [],
                                        "user_id": [],
                                    })

if "user_id" not in st.session_state:
    st.session_state.user_id = 0

if "book_data" not in st.session_state:
    st.session_state.book_data = pd.read_csv("Data/final_books.csv")
    # st.session_state.book_data = st.session_state.book_data[['book_id', 'title', 'image_url']]

if "predict" not in st.session_state:
    st.session_state.predict = []

if "model" not in st.session_state:
    st.session_state.model = NeuralMatrixFactoration(
                                weight_path="weights/new_non_biases_NeuMF.weights.h5",
                                interaction_file="Data/interaction.csv",
                                book_file="Data/final_books.csv"
                            )

if "ratings" not in st.session_state:
    st.session_state.ratings = {}

if "old_ratings" not in st.session_state:
    st.session_state.old_ratings = {}

if "random_books" not in st.session_state:
    st.session_state.random_books = st.session_state.book_data.sample(10)

if "view" not in st.session_state:
    st.session_state.view = "login"

if "new_user" not in st.session_state:
    st.session_state.new_user = False

if "new_data" not in st.session_state:
    st.session_state.new_data = pd.DataFrame(columns=['user_id', 'book_id', 'rating'])

def check_user_exists(username):
    user = st.session_state.users[st.session_state.users['username'] == username]
    if not user.empty:
        return True
    else:
        return False

def login():
    st.title("📚 Book Recommender System")
    st.subheader("Login or Sign Up")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            check = check_user_exists(username)
            if not check:
                st.error("User not found")
            else:
                if password != st.session_state.users[st.session_state.users['username'] == username]['password'].values[0]:
                    st.error("Incorrect password")
                else:
                    st.success("Login successful!")
                    st.session_state.current_user = st.session_state.users[st.session_state.users['username'] == username]
                    st.session_state.user_id = st.session_state.users[st.session_state.users['username'] == username]['user_id'].values[0]
                    st.session_state.old_ratings = st.session_state.model.data[st.session_state.model.data['user_id'] == st.session_state.user_id]['rating'].values
                    st.session_state.old_ratings = {k: v for k, v in zip(st.session_state.model.data[st.session_state.model.data['user_id'] == st.session_state.user_id]['book_id'].values, st.session_state.old_ratings)}
                    st.session_state.random_books = st.session_state.book_data[~st.session_state.book_data['book_id'].isin(st.session_state.old_ratings.keys())].sample(10)
                    if len(st.session_state.old_ratings) < 5:
                        st.session_state.view = "rate"
                    else:
                        st.session_state.view = "recommend"
                    if not st.session_state.new_user:
                        st.session_state.predict = st.session_state.model.get_top_k_recommendations(st.session_state.user_id, k=1000)
                        # highest_rated_books = st.session_state.model.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
                        # most_rated_books = st.session_state.model.data.groupby('book_id')['rating'].mean().sort_values(ascending=False).index[:50]
                        #eliminate the books that are highest rated or most rated
                        # st.session_state.predict = [book for book in st.session_state.predict if book[0] not in highest_rated_books and book[0] not in most_rated_books]
                        # st.session_state.predict = [book for book in st.session_state.predict if st.session_state.model.data[st.session_state.model.data['book_id']==book[0]]['rating'].mean() < 4.2]
                    st.rerun()
            # if user and user["password"] == password:
            #     st.session_state.current_user = username
            #     st.success("Login successful!")
            #     st.session_state.user_id = max(st.session_state.model.data['user_id'].unique()) + 1
            #     st.rerun()
            # else:
            #     st.error("Invalid credentials")

    with col2:
        if st.button("Sign Up"):
            st.session_state.view = "signup"
            # if username in st.session_state.users:
            #     st.error("Username already exists")
            # else:
            #     st.session_state.users[username] = {
            #         "password": password,
            #         "user_id": len(st.session_state.users),
            #         "ratings": {}
            #     }
            #     st.session_state.current_user = username
            #     st.success("Sign-up successful!")
            st.rerun()

def signup():
    st.subheader("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    gender = st.radio(
            f"Gender",
                options=["Male", "Female"]
            )
    if st.button("Sign Up"):
        check = check_user_exists(username)
        if check:
            st.error("User already exists")
        elif username == "" or password == "":
            st.error("Please fill in all fields")
        else:
            st.session_state.user_id = max(st.session_state.model.data['user_id'].unique()) + 1
            st.session_state.current_user = pd.DataFrame({
                                                "username": [username],
                                                "password": [password],
                                                "user_id": [st.session_state.user_id],
                                            })
            st.session_state.users = pd.concat([st.session_state.users, st.session_state.current_user], ignore_index=True)
            # st.session_state.users.to_csv("data/user_data.csv")
            new_user = st.session_state.current_user.values[0].tolist()
            new_user = [st.session_state.user_id] + new_user
            with open('Data/user_data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_user)
            st.success("Sign-up successful!")
            st.session_state.new_user = True
            st.session_state.view = "login"
            st.rerun()

def rating_screen():
    st.title("📖 Rate at least 5 books")
    st.subheader("To get recommendations, rate at least 5 books")

    # user_data = st.session_state.users[st.session_state.current_user]
    user_data = st.session_state.model.data[st.session_state.model.data['user_id'] == st.session_state.user_id]

    search_term = st.text_input("🔍 Search for a book title").strip().lower()
    
    if search_term:
        books_to_show = st.session_state.book_data[st.session_state.book_data['title'].str.contains(search_term, case=False)]
        #take 10 random books from the filtered data that have not been rated
        st.session_state.random_books = st.session_state.book_data[~st.session_state.book_data['book_id'].isin(st.session_state.old_ratings.keys())].sample(10)
    else:
        # st.session_state.random_books = st.session_state.book_data[~st.session_state.book_data['book_id'].isin(st.session_state.ratings.keys())].sample(10)
        books_to_show = st.session_state.random_books
    
    for _, book in books_to_show.iterrows():
        st.image(book["image_url"], width=100)
        st.markdown(f"<div style='font-size:18px; font-weight:bold;'>⭐ {book['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:16px'>Genre: {book['most_tagged']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:15px'>Average Rating: {st.session_state.model.data[st.session_state.model.data['book_id']==book['book_id']]['rating'].mean():.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px; color:gray;'>{book['description']}</div>", unsafe_allow_html=True)
        book_id = book["book_id"]
        initial_rating = st.session_state.old_ratings.get(book_id, 0)
        slider_key = f"rating_{book['book_id']}"
        rating_value = st.slider(
                            f"Rate {book['title']}", 
                            min_value=0, 
                            max_value=5, 
                            value=initial_rating,
                            step=1,
                            key=slider_key
                        )
        st.session_state.ratings[book["book_id"]] = rating_value
    
    if st.button("Reload Books"):
        del st.session_state.random_books
        st.session_state.random_books = st.session_state.book_data[~st.session_state.book_data['book_id'].isin(list(st.session_state.ratings.keys())+list(st.session_state.old_ratings.keys()))].sample(10)
        st.session_state.view = "rate"
        st.rerun()
    if st.button("Submit Ratings"):
        rated_books = {st.session_state.book_data[st.session_state.book_data["book_id"]==k]['title'].values[0]: v for k, v in st.session_state.ratings.items() if v > 0}
        if len(rated_books) < 5:
            st.warning("⚠️ Please rate at least 5 books.")
        else:
            book_ids = st.session_state.book_data[st.session_state.book_data['book_id'].isin(st.session_state.ratings.keys())]['book_id'].values
            ratings_values = st.session_state.ratings.values()
            new_data = pd.DataFrame({
                'user_id': [st.session_state.user_id] * len(book_ids),
                'book_id': book_ids,
                'rating': ratings_values
            })
            st.session_state.new_data = new_data[new_data['rating'] > 0]
            with open('Data/interaction.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                for i, data in st.session_state.new_data.iterrows():
                    writer.writerow([st.session_state.model.data.shape[0]+i+1]+data.values.tolist())
    if st.button("Get Recommendations"):
        st.success("Thank you! Generating recommendations...")
        if st.session_state.new_data.shape[0] > 0:
            st.session_state.model.update_user(st.session_state.new_user, st.session_state.new_data, learning_rate=0.001, epochs=15, batch_size=4, save_path='weights/new_non_biases_NeuMF.weights.h5')
        st.session_state.predict = st.session_state.model.get_top_k_recommendations(st.session_state.user_id, k=20)
        st.session_state.predict = make_recommendations()
        st.session_state.view = "recommend"
        st.rerun()
    
def make_recommendations():
    # st.session_state.predict = st.session_state.model.get_top_k_recommendations(st.session_state.user_id, k=100)
    rated_books = []
    for i in range(len(st.session_state.predict)):
        if st.session_state.predict[i][0] in st.session_state.ratings.keys():
            rated_books.append(st.session_state.predict[i])
    for i in rated_books:
        st.session_state.predict.pop(st.session_state.predict.index(i))
    st.session_state.predict = st.session_state.predict[:20]
    return st.session_state.predict

def recommendation_screen():
    book_pairs = [st.session_state.predict[i:i+2] for i in range(0, len(st.session_state.predict), 2)]

    for pair in book_pairs:
        col1, col2 = st.columns(2)

        with col1:
            if len(pair) > 0:
                book_id = pair[0][0]
                book_title = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'title'].values[0]
                book_image = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'image_url'].values[0]
                book_genre = st.session_state.book_data.loc[st.session_state.book_data['book_id'] == book_id, 'most_tagged'].values[0]
                book_description = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'description'].values[0]
                book_avg_rating = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'rating'].mean()
                with st.container():
                    img_col, text_col = st.columns([1, 3])
                    with img_col:
                        st.image(book_image, width=100)
                    with text_col:
                        st.markdown(f"<div style='font-size:18px; font-weight:bold;'>⭐ {book_title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:16px'>Genre: {book_genre}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:15px'>Average Rating: {book_avg_rating:.2f}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:14px; color:gray;'>{book_description}</div>", unsafe_allow_html=True)

        with col2:
            if len(pair) > 1:
                book_id = pair[1][0]
                book_title = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'title'].values[0]
                book_image = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'image_url'].values[0]
                book_genre = st.session_state.book_data.loc[st.session_state.book_data['book_id'] == book_id, 'most_tagged'].values[0]
                book_description = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'description'].values[0]
                book_avg_rating = st.session_state.model.data.loc[st.session_state.model.data['book_id'] == book_id, 'rating'].mean()
                with st.container():
                    img_col, text_col = st.columns([1, 3])
                    with img_col:
                        st.image(book_image, width=100)
                    with text_col:
                        st.markdown(f"<div style='font-size:18px; font-weight:bold;'>⭐ {book_title}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:16px'>Genre: {book_genre}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:15px'>Average Rating: {book_avg_rating:.2f}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:14px; color:gray;'>{book_description}</div>", unsafe_allow_html=True)

    st.subheader(f"Hi {st.session_state.current_user["username"].values[0]}, you might like:")
    st.session_state.new_user = False
    if st.button("🔄 Go back and rate more books"):
        st.session_state.view = "rate"
        st.session_state.random_books = st.session_state.book_data.sample(10)
        st.rerun()


    if st.button("Logout"):
        keys_to_clear = [
            "current_user", "user_id", "ratings", "predict",
            "random_books", "view", "new_user", "new_data", 
            "old_ratings"
        ]
        st.session_state.view = "login"
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        gc.collect()
        st.rerun()

def main():
    if st.session_state.view == "login":
        login()
    elif st.session_state.view == "signup":
        signup()
    else:
        user_data = st.session_state.model.data[st.session_state.model.data['user_id'] == st.session_state.user_id]
        if user_data.shape[0] == 0 or (len(st.session_state.old_ratings)+len(st.session_state.ratings)) < 5 or st.session_state.view == "rate":
            rating_screen()
        else:
            # st.session_state.predict = st.session_state.model.get_top_k_recommendations(st.session_state.user_id, k=100)
            st.session_state.predict = make_recommendations()
            recommendation_screen()

if __name__ == "__main__":
    main()
