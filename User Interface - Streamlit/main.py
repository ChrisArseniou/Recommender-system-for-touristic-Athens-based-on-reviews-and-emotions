import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from functions import *
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import operator

st.set_page_config(
    page_title="Athens Trip Recommender",
    page_icon="athens.png",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("Athens Trip Recommender")

what_to_recommend =st.sidebar.selectbox("Recommenders",["Airlines","Accommodation", 'Restaurant', 'Activities'])

st.markdown("""
 <style>
 .big-font {
     font-size:30px !important;
 }
 </style>
 """, unsafe_allow_html=True)

emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

if(what_to_recommend == 'Restaurant'):
    st.markdown('<p class="big-font">Restaurant Recommendations</p>', unsafe_allow_html=True)

    option = st.selectbox(
        "Help me show you my recommendations!",
        ["I know what i am looking for!", "Recommend based on my past."]
    )

    if (option == 'I know what i am looking for!'):
        text = st.text_area("What are you looking for?", height=200, max_chars=300)
        if st.button("Find Restaurant") and (text or len(text) != 0):
            with st.spinner(f"Finding Restaurant..."):
                df = pd.read_csv('Datasets/Restaurants/restaurants_for_keywords_recommendation.csv')
                recommendations = get_recommendations_with_emotion(df, text)
                st.write(recommendations[:5])
        else:
            st.warning("Please enter the text")
    else:
        username = st.text_input('Username')
        password = st.text_input('Password', type="password")

        if st.button("Find Restaurant") and (password or len(password) != 0) and int(username) < 1006 and int(username) > 0:
            with st.spinner(f"Finding Restaurant..."):
                # Finding restaurants that provides the user's most felt emotion the most
                df_emotions = pd.read_csv('Datasets/Restaurants/restaurants_for_common_emotions.csv')
                df_emotions.drop(["Unnamed: 0"], axis=1, inplace=True)
                user_comments = df_emotions[df_emotions['UserId'] == int(username)]
                user_positive_comments = user_comments[user_comments['Rating'] > 3]
                already_visited = user_comments["Name"].values
                user_positive_comments = user_positive_comments.groupby("Emotion").count()
                user_positive_comments = user_positive_comments.sort_values(by=['Emotion'], ascending=True)

                if len(user_positive_comments) == 0:
                    st.warning('No recommendations!')
                else:
                    most_felt_emotion = user_positive_comments.index.values[0]
                    possible_options = df_emotions[df_emotions['Emotion'] == most_felt_emotion]
                    recommends_df = possible_options.groupby("Name")['Emotion'].count().to_frame()
                    recommends_df = recommends_df.sort_values(by=['Emotion'], ascending=False)
                    possible_recommendations = recommends_df.index.values
                    recommendations_from_emotion = []
                    for i in possible_recommendations:
                        if i not in already_visited:
                            recommendations_from_emotion.append(i)

                    # SVD Analysis
                    df = pd.read_csv('Datasets/Restaurants/restaurants_for_SVD.csv')
                    pivot = pd.pivot_table(df, index='UserId', columns=['Name'], values='Rating').fillna(0)
                    pivot_mat = pivot.values
                    user_id = list(pivot.index)
                    sparse_matrix = csr_matrix(pivot_mat)
                    factor_n = 15
                    U, sigma, V = svds(sparse_matrix, k=factor_n)
                    sigma = np.diag(sigma)
                    pred_rating = np.dot(np.dot(U, sigma), V)
                    pred_rating_n = (pred_rating - pred_rating.min()) / (pred_rating.max() - pred_rating.min()) # κανονικοποίηση
                    pred_df = pd.DataFrame(pred_rating_n, columns=pivot.columns, index=user_id).transpose()
                    restaurants = pd.read_csv('Datasets/Restaurants/restaurants.csv')
                    ratings = pd.read_csv('Datasets/Restaurants/ratings.csv')
                    df_meta = restaurants.merge(ratings, how="left", on="RestaurantId")
                    df_meta = df_meta.astype({"UserId": str, "RestaurantId": str})
                    df = df.astype({"UserId": str})
                    recommendations_from_SVD = recommender(int(username), pred_df, df, df_meta)

                    score_dict = {}

                    for index, row in recommendations_from_SVD.iterrows():
                        name = row['Name']
                        recStrength = row['recStrength']
                        if (len(recommendations_from_emotion) >= 3):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                            elif (name == recommendations_from_emotion[1]):
                                recStrength = recStrength + 0.1 * recStrength
                            elif (name == recommendations_from_emotion[2]):
                                recStrength = recStrength + 0.05 * recStrength
                        elif (len(recommendations_from_emotion) == 2):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                            elif (name == recommendations_from_emotion[1]):
                                recStrength = recStrength + 0.1 * recStrength
                        elif (len(recommendations_from_emotion) == 1):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                        score_dict[index] = recStrength

                    sorted_scores = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

                    counter = 0
                    # δημιουργία νέου dataframe για την αποθήκευση των recommendations
                    recommendations = pd.DataFrame(columns=('Restaurant', 'score'))
                    # τα reviews με τα υψηλότερα cosign similarity
                    for i in sorted_scores:
                        recommendations = recommendations.append({'Restaurant': recommendations_from_SVD.iloc[i[0]]['Name'],
                                                                  'score': i[1]}, ignore_index=True)

                        counter += 1
                        if counter > 10:
                            break

                    st.write(recommendations[:5])
        else:
            st.warning("Enter correct information")


elif(what_to_recommend == 'Airlines'):
    st.markdown('<p class="big-font">Airlines Recommendation</p>', unsafe_allow_html=True)

    option = st.selectbox(
        "Help me show you my recommendations!",
        ["I know what i am looking for!", "Recommend based on my past."]
    )

    if (option == 'I know what i am looking for!'):
        text = st.text_area("What are you looking for?", height=200, max_chars=300)
        if st.button("Find Airline") and (text or len(text) != 0):
            with st.spinner(f"Finding Airline..."):
                df = pd.read_csv('Datasets/Airlines/airlines_for_keywords_recommendation.csv')
                recommendations = get_recommendations_with_emotion(df, text)
                st.write(recommendations[:5])
        else:
            st.warning("Please enter the text")
    else:
        username = st.text_input('Username')
        password = st.text_input('Password', type="password")

        if st.button("Find Airline") and (password or len(password) != 0) and int(username) < 2006 and int(
                username) > 0:
            with st.spinner(f"Finding Airline..."):
                # Finding airlines that provides the user's most felt emotion the most
                df_emotions = pd.read_csv('Datasets/Airlines/reviews_with_emotions_full.csv')
                df_emotions.drop(["Unnamed: 0"], axis=1, inplace=True)
                user_comments = df_emotions[df_emotions['UserId'] == int(username)]
                user_positive_comments = user_comments[user_comments['Rating'] > 3]
                already_visited = user_comments["Name"].values
                user_positive_comments = user_positive_comments.groupby("Emotion").count()
                user_positive_comments = user_positive_comments.sort_values(by=['Emotion'], ascending=True)

                if len(user_positive_comments) == 0:
                    st.warning('No recommendations!')
                else:
                    most_felt_emotion = user_positive_comments.index.values[0]
                    possible_options = df_emotions[df_emotions['Emotion'] == most_felt_emotion]
                    recommends_df = possible_options.groupby("Name")['Emotion'].count().to_frame()
                    recommends_df = recommends_df.sort_values(by=['Emotion'], ascending=False)
                    possible_recommendations = recommends_df.index.values
                    recommendations_from_emotion = []
                    for i in possible_recommendations:
                        if i not in already_visited:
                            recommendations_from_emotion.append(i)

                    # SVD Analysis
                    df = pd.read_csv('Datasets/Airlines/airlines_for_keywords_recommendation.csv')
                    pivot = pd.pivot_table(df, index='UserId', columns=['Name'], values='Rating').fillna(0)
                    pivot_mat = pivot.values
                    user_id = list(pivot.index)
                    sparse_matrix = csr_matrix(pivot_mat)
                    factor_n = 8
                    U, sigma, V = svds(sparse_matrix, k=factor_n)
                    sigma = np.diag(sigma)
                    pred_rating = np.dot(np.dot(U, sigma), V)
                    pred_rating_n = (pred_rating - pred_rating.min()) / (
                                pred_rating.max() - pred_rating.min())  # κανονικοποίηση
                    pred_df = pd.DataFrame(pred_rating_n, columns=pivot.columns, index=user_id).transpose()

                    airlines = pd.read_csv('Datasets/Airlines/airlines.csv')
                    ratings = pd.read_csv('Datasets/Airlines/ratings.csv')
                    df_meta = airlines.merge(ratings, how="left", on="AirlineId")
                    df_meta = df_meta.astype({"UserId": str, "AirlineId": str})
                    df = df.astype({"UserId": str})
                    recommendations_from_SVD = airline_recommender(int(username), pred_df, df, df_meta)

                    score_dict = {}

                    for index, row in recommendations_from_SVD.iterrows():
                        name = row['Name']
                        recStrength = row['recStrength']
                        if (len(recommendations_from_emotion) >= 3):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                            elif (name == recommendations_from_emotion[1]):
                                recStrength = recStrength + 0.1 * recStrength
                            elif (name == recommendations_from_emotion[2]):
                                recStrength = recStrength + 0.05 * recStrength
                        elif (len(recommendations_from_emotion) == 2):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                            elif (name == recommendations_from_emotion[1]):
                                recStrength = recStrength + 0.1 * recStrength
                        elif (len(recommendations_from_emotion) == 1):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                        score_dict[index] = recStrength

                    sorted_scores = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

                    counter = 0
                    # δημιουργία νέου dataframe για την αποθήκευση των recommendations
                    recommendations = pd.DataFrame(columns=('Airline', 'score'))
                    # τα reviews με τα υψηλότερα cosign similarity
                    for i in sorted_scores:
                        recommendations = recommendations.append({'Airline': recommendations_from_SVD.iloc[i[0]]['Name'],
                                                                  'score': i[1]}, ignore_index=True)

                        counter += 1
                        if counter > 10:
                            break

                    st.write(recommendations[:5])
        else:
            st.warning("Enter correct information")



elif(what_to_recommend == 'Accommodation'):
    st.markdown('<p class="big-font">Hotels Recommendation</p>', unsafe_allow_html=True)

    option = st.selectbox(
        "Help me show you my recommendations!",
        ["I know what i am looking for!", "Recommend based on my past."]
    )

    if (option == 'I know what i am looking for!'):
        text = st.text_area("What are you looking for?", height=200, max_chars=300)

        if st.button("Find Hotel") and (text or len(text) != 0):
            with st.spinner(f"Finding Hotel..."):
                hotel_recommendations = hotel_recommend_with_knn([text])
                hotel_recommendations_df = pd.DataFrame(hotel_recommendations, columns = ['Hotels'])
                st.write(hotel_recommendations_df[:5])
        else:
            st.warning("Please enter the text")

    else:
        username = st.text_input('Username')
        password = st.text_input('Password', type="password")

        if st.button("Find Hotel") and (password or len(password) != 0) and int(username) < 2006 and int(
                username) > 0:
            with st.spinner(f"Finding Hotel..."):

                # Finding hotels that provides the user's most felt emotion the most
                df_emotions = pd.read_csv('Datasets/Hotels/reviews_with_emotions_full.csv')
                df_emotions.drop(["Unnamed: 0"], axis=1, inplace=True)
                user_comments = df_emotions[df_emotions['UserId'] == int(username)]
                user_positive_comments = user_comments[user_comments['Rating'] > 3]
                already_visited = user_comments["Name"].values
                user_positive_comments = user_positive_comments.groupby("Emotion").count()
                user_positive_comments = user_positive_comments.sort_values(by=['Emotion'], ascending=True)

                if len(user_positive_comments) == 0:
                    st.warning('No recommendations!')
                else:

                    most_felt_emotion = user_positive_comments.index.values[0]
                    possible_options = df_emotions[df_emotions['Emotion'] == most_felt_emotion]
                    recommends_df = possible_options.groupby("Name")['Emotion'].count().to_frame()
                    recommends_df = recommends_df.sort_values(by=['Emotion'], ascending=False)
                    possible_recommendations = recommends_df.index.values
                    recommendations_from_emotion = []
                    for i in possible_recommendations:
                        if i not in already_visited:
                            recommendations_from_emotion.append(i)

                    # SVD Analysis
                    df = pd.read_csv('Datasets/Hotels/hotels_for_keywords_recommendation.csv')
                    pivot = pd.pivot_table(df, index='UserId', columns=['Name'], values='Rating').fillna(0)
                    pivot_mat = pivot.values
                    user_id = list(pivot.index)
                    sparse_matrix = csr_matrix(pivot_mat)
                    factor_n = 8
                    U, sigma, V = svds(sparse_matrix, k=factor_n)
                    sigma = np.diag(sigma)
                    pred_rating = np.dot(np.dot(U, sigma), V)
                    pred_rating_n = (pred_rating - pred_rating.min()) / (
                    pred_rating.max() - pred_rating.min())  # κανονικοποίηση
                    pred_df = pd.DataFrame(pred_rating_n, columns=pivot.columns, index=user_id).transpose()

                    hotels = pd.read_csv('Datasets/Hotels/hotels.csv')
                    ratings = pd.read_csv('Datasets/Hotels/ratings.csv')
                    df_meta = hotels.merge(ratings, how="left", on="HotelId")
                    df_meta = df_meta.astype({"UserId": str, "HotelId": str})
                    df = df.astype({"UserId": str})
                    recommendations_from_SVD = hotel_recommender(int(username), pred_df, df, df_meta)

                    score_dict = {}

                    for index, row in recommendations_from_SVD.iterrows():
                        name = row['Name']
                        recStrength = row['recStrength']
                        if (len(recommendations_from_emotion) >= 3):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                            elif (name == recommendations_from_emotion[1]):
                                recStrength = recStrength + 0.1 * recStrength
                            elif (name == recommendations_from_emotion[2]):
                                recStrength = recStrength + 0.05 * recStrength
                        elif (len(recommendations_from_emotion) == 2):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                            elif (name == recommendations_from_emotion[1]):
                                recStrength = recStrength + 0.1 * recStrength
                        elif (len(recommendations_from_emotion) == 1):
                            if (name == recommendations_from_emotion[0]):
                                recStrength = recStrength + 0.2 * recStrength
                        score_dict[index] = recStrength

                    sorted_scores = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

                    counter = 0
                    # δημιουργία νέου dataframe για την αποθήκευση των recommendations
                    recommendations = pd.DataFrame(columns=('Hotels', 'score'))
                    # τα reviews με τα υψηλότερα cosign similarity
                    for i in sorted_scores:
                        recommendations = recommendations.append({'Hotels': recommendations_from_SVD.iloc[i[0]]['Name'],
                                                                  'score': i[1]}, ignore_index=True)

                        counter += 1
                        if counter > 10:
                            break

                    st.write(recommendations[:5])
        else:
            st.warning("Enter correct information")

elif what_to_recommend == 'Activities':
    st.markdown('<p class="big-font">Activities Recommendation</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    print("Works")

