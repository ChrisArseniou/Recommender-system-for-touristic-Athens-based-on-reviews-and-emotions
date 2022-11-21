#@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
import re
import pandas as pd
import numpy as np
import math
from collections import Counter
from transformers import pipeline
import operator
from joblib import load
import pickle
import sklearn

def cosine_similarity_of(text1, text2):
    # taking words
    first = re.compile(r"[\w']+").findall(text1)
    second = re.compile(r"[\w']+").findall(text2)

    # creating dictionary from each word and count
    vector1 = Counter(first)
    vector2 = Counter(second)

    # changing vectors to sets to find comman words with intersection
    common = set(vector1.keys()).intersection(set(vector2.keys()))

    dot_product = 0.0

    for i in common:
        # getting amount of each common word for vectors and multiply them also add them
        dot_product += vector1[i] * vector2[i]

    squared_sum_vector1 = 0.0
    squared_sum_vector2 = 0.0

    # getting summation of word counts for every vector
    for i in vector1.keys():
        squared_sum_vector1 += vector1[i] ** 2

    for i in vector2.keys():
        squared_sum_vector2 += vector2[i] ** 2

    # calculating magnitude of vector with squared sums
    magnitude = math.sqrt(squared_sum_vector1) * math.sqrt(squared_sum_vector2)

    if not magnitude:
        return 0.0
    else:
        return float(dot_product) / magnitude


def calculate_final_score(cos_sim, w):
# ανάθεση βάρους στο cosine similarity
    effect = (cos_sim / 100) * w
    return cos_sim + effect


def get_rating_weight_with_threshold(rating, count, threshold, Q):
    # rating effect
    w = (2 * Q / 5) * rating - Q
    # multiplier effect
    M = math.exp((-threshold * 0.68) / count)

    return w * M

def get_recommendations_with_emotion(df, keywords):

    score_dict = {}
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion_from_keywords = emotion(keywords)[0]['label']
    print('Emotion is ' + emotion_from_keywords)

    for index, row in df.iterrows():
        cs = cosine_similarity_of(str(row['word_without_stop']), keywords)

        rating = row['Rating']
        rating_count = row['Restaurant_Count']
        emotion_from_comment = row['Emotion']

        if row['positive'] == True:
            pos_rat = (7 + rating) / 2
        else:
            pos_rat = rating / 2

        threshold = 110
        # I choose Q as 10 to not to change weight too much
        rating_contribution = get_rating_weight_with_threshold(pos_rat, rating_count, threshold, 10)

        score_from_similarity = calculate_final_score(cs, rating_contribution)

        # Αν το συναίσθημα από τα keywords ισοδυναμεί με το συναίσθημα του σχολίου τότε το score αυξάνεται κατά 10%
        if (emotion_from_comment == emotion_from_keywords):
            last_score = score_from_similarity + 0.2 * score_from_similarity
        else:
            last_score = score_from_similarity

        score_dict[index] = last_score

    # sort τα σχόλια
    sorted_scores = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)

    counter = 0

    # δημιουργία νέου dataframe για την αποθήκευση των recommendations
    result = pd.DataFrame(columns=('Name', 'score'))

    # τα reviews με τα υψηλότερα cosign similarity
    for i in sorted_scores:
        result = result.append({'Name': df.iloc[i[0]]['Name'],'score': i[1]}, ignore_index=True)
        counter += 1
        if counter > 20:
            break

    result = result.drop_duplicates(subset=['Name'], keep='first')
    result = result.reset_index(drop=True)
    return result

def recommend_restaurants(user_id, pred_df, items_df, restaurants_to_ignore=[], top_list=20, verbose=False):
    # η συνάρτηση βρίσκει τα πιο κοντινά εστιατόρεια
    sorted_user_predictions = pred_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

    recommendations_df = sorted_user_predictions[
        ~sorted_user_predictions['Name'].isin(restaurants_to_ignore)].sort_values(by='recStrength',
                                                                                      ascending=False).head(top_list)
    print(recommendations_df)
    return recommendations_df


def recommender(user_id,pred_df,real_df,df_meta):

    restaurants_to_ignore = real_df[real_df['UserId'] == str(user_id)]['Name'].values
    recommend = recommend_restaurants(user_id, pred_df,real_df, restaurants_to_ignore)
    df_user= real_df.loc[real_df['UserId'] == user_id]
    new_df = df_user.merge(recommend, how = 'outer', left_on = 'Name', right_on = 'Name')
    rec_df = new_df.loc[new_df['Rating'].isnull()==True]
    df_rec_t = rec_df.loc[:, ['Name', 'recStrength']]
    df_last = pd.merge(df_rec_t,df_meta[['RestaurantId','Name']],on=['Name'], how='left')
    df_last = df_last.drop_duplicates(subset=["Name", "RestaurantId"], keep='first')
    df_last = df_last.reset_index(drop=True)
    return df_last



def recommend_airlines(user_id, pred_df, items_df, airlines_to_ignore=[], top_list=20, verbose=False):
    sorted_user_predictions = pred_df[user_id].sort_values(ascending=False).reset_index().rename(
        columns={user_id: 'recStrength'})

    recommendations_df = sorted_user_predictions[
        ~sorted_user_predictions['Name'].isin(airlines_to_ignore)].sort_values(by='recStrength',
                                                                                   ascending=False).head(top_list)
    return recommendations_df


def airline_recommender(user_id,pred_df,real_df,df_meta):

    airlines_to_ignore = real_df[real_df['UserId'] == str(user_id)]['Name'].values
    recommend = recommend_airlines(user_id, pred_df,real_df, airlines_to_ignore)
    df_user= real_df.loc[real_df['UserId'] == user_id]
    new_df = df_user.merge(recommend, how = 'outer', left_on = 'Name', right_on = 'Name')
    rec_df = new_df.loc[new_df['Rating'].isnull()==True]
    df_rec_t = rec_df.loc[:, ['Name', 'recStrength']]
    df_last = pd.merge(df_rec_t,df_meta[['AirlineId','Name']],on=['Name'], how='left')
    df_last = df_last.drop_duplicates(subset=["Name", "AirlineId"], keep='first')
    df_last = df_last.reset_index(drop=True)
    return df_last

def hotel_recommend_with_knn(text):
  pipeline = load("Models/Hotels/text_classification_for_hotels.joblib")
  pickle_in = open("Models/Hotels/dict_for_hotels.pickle","rb")
  example_dict = pickle.load(pickle_in)
  out_pred = pipeline.predict(text)
  list_recomm = [id for id, pred in example_dict.items() if pred == out_pred][:5]
  return list_recomm


def recommend_hotels(user_id, pred_df, items_df, hotels_to_ignore=[], top_list=20, verbose=False):
    sorted_user_predictions = pred_df[user_id].sort_values(ascending=False).reset_index().rename(
        columns={user_id: 'recStrength'})

    recommendations_df = sorted_user_predictions[~sorted_user_predictions['Name'].isin(hotels_to_ignore)].sort_values(
        by='recStrength', ascending=False).head(top_list)
    print(recommendations_df)
    return recommendations_df

def hotel_recommender(user_id,pred_df,real_df,df_meta):

    restaurants_to_ignore = real_df[real_df['UserId'] == str(user_id)]['Name'].values
    recommend = recommend_hotels(user_id, pred_df,real_df, restaurants_to_ignore)
    df_user= real_df.loc[real_df['UserId'] == user_id]
    new_df = df_user.merge(recommend, how = 'outer', left_on = 'Name', right_on = 'Name')
    rec_df = new_df.loc[new_df['Rating'].isnull()==True]
    df_rec_t = rec_df.loc[:, ['Name', 'recStrength']]
    df_last = pd.merge(df_rec_t,df_meta[['HotelId','Name']],on=['Name'], how='left')
    df_last = df_last.drop_duplicates(subset=["Name", "HotelId"], keep='first')
    df_last = df_last.reset_index(drop=True)
    return df_last