import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url:str = os.getenv("SUPABASE_ENDPOINT")
key:str = os.getenv("SUPABASE_API_KEY")
tbl:str = os.getenv("SUPABASE_TBL")

def fetch_data():
    with httpx.Client() as client:  # Use httpx.Client instead of httpx.AsyncClient
        response = client.get(
            f'{url}/rest/v1/{tbl}',
            headers={'apikey': key},
            params={'select': 'conJoin,No'}  # Adjust the query parameters as needed
        )
        response.raise_for_status()  # Raise an error for unsuccessful HTTP responses
        data = response.json()
    return data


def load_data():
    data = fetch_data()
    data = [item for item in data if item['conJoin'] is not None]
    loaded_data = pd.DataFrame(data)
    print("Loaded===>",loaded_data)
    return loaded_data


# def loop(data):
#     # print("data", data)
#     df = pd.DataFrame(data, columns=['No','conJoin'])
    
#     flatten_data = []
#     for _, row in df.iterrows():
#         for conjoin_value in row['conJoin']:    
#             flatten_data.append([row['No'],conjoin_value])
#     flatten_data = pd.DataFrame(flatten_data, columns=['No','conJoin'])
#     print("flatten data::>>\n\n\n",flatten_data)
#     return flatten_data

def loop(data):
    if data is None:
        return pd.DataFrame(columns=['No', 'conJoin'])

    flatten_data = []
    for _, row in pd.DataFrame(data).iterrows():
        no_value = row['No']
        conjoin_values = row['conJoin']

        for conjoin_value in conjoin_values:
            flatten_data.append({'No': no_value, 'conJoin': conjoin_value})

    flatten_df = pd.DataFrame(flatten_data, columns=['No', 'conJoin'])
    return flatten_df

def create_user_item_matrix(data):
    user_item_matrix = data.groupby(['No', 'conJoin']).size().unstack(fill_value=0)
    # print("User-item matrix:\n", user_item_matrix)
    return user_item_matrix

def collaborative_filtering_cosine_similarity(user_item_matrix):
    # Check if the user_item_matrix is empty
    if user_item_matrix.empty:
        print("User-item matrix is empty.")
        return None

    # Check if the user_item_matrix has any NaN values
    if user_item_matrix.isna().any().any():
        print("User-item matrix contains NaN values. Please fill or handle missing values.")
        return None

    # Normalize the user-item matrix before computing cosine similarity
    user_item_matrix_normalized = normalize(user_item_matrix)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(user_item_matrix_normalized, user_item_matrix_normalized)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    print("cosine_sim_df:\n\n" , cosine_sim_df)
    return cosine_sim_df

# def kmeans(user_item_matrix):
    # # Check if the user_item_matrix is empty
    # if user_item_matrix.empty:
    #     print("User-item matrix is empty.")
    #     return None

    # # Check if the user_item_matrix has any NaN values
    # if user_item_matrix.isna().any().any():
    #     print("User-item matrix contains NaN values. Please fill or handle missing values.")
    #     return None

    # # Drop any non-numeric columns if present
    # user_item_matrix_numeric = user_item_matrix.select_dtypes(include=['number'])

    # # Check if the user_item_matrix_numeric is empty after dropping non-numeric columns
    # if user_item_matrix_numeric.empty:
    #     print("No numeric columns found in the user-item matrix.")
    #     return None

    # # Normalize the user-item matrix
    # user_item_matrix_normalized = normalize(user_item_matrix_numeric)

    # # Perform KMeans clustering
    # kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
    # user_item_matrix['cluster'] = kmeans.fit_predict(user_item_matrix_normalized)

    return user_item_matrix

def recommend(user_id, user_item_matrix, cosine_sim):
    # Get the activities that the target user has already participated in
    # change 0 to 1 to change from show what you not selected and show what you selected
    user_activities_participated = user_item_matrix.columns[user_item_matrix.loc[user_id].eq(1)]

    # Extract recommendations based on cosine similarity
    user_similarity = cosine_sim.loc[user_id]
    similar_users = user_similarity.sort_values(ascending=False).index

    # Exclude the current user from recommendations
    similar_users = similar_users[similar_users != user_id]

    # Filter activities based on the threshold
    recommended_activities = user_item_matrix.loc[similar_users, user_activities_participated].sum()
    # recommended_activities = recommended_activities[recommended_activities >= threshold]
    # Sort recommendations by the sum of participation scores
    recommendations_sorted = recommended_activities.sort_values(ascending=False)
    print("recommend:",recommendations_sorted)
    # Convert recommendations to a list of dictionaries
    recommendations_list = [
        {"type_id": int(activity_id), "score": int(score)}
        for activity_id, score in recommendations_sorted.items()
    ]

    return recommendations_list



@app.get("/")
async def root():


    return {"AI": url}

@app.get("/user_id/{user_id}")
async def read_user_data(user_id: int):
    data = load_data()
    flatted = loop(data)
    user_item_matrix = create_user_item_matrix(flatted)
    cosine_sim = collaborative_filtering_cosine_similarity(user_item_matrix)

    if user_id not in user_item_matrix.index:
        raise HTTPException(status_code=404, detail="User not found")

    # Remove the kmeans call
    # user_item_matrix = kmeans(user_item_matrix)

    recommendations = recommend(user_id, user_item_matrix, cosine_sim)

    output_str = f"Top Recommendations for User {user_id}:\n"
    errorMessages = [{
        "code": 404,
        "message": "No recommendations found for the user."
    }]

    # Check if recommendations is not empty
    if recommendations:
        print(recommendations)
        return recommendations
    else:
        return errorMessages


