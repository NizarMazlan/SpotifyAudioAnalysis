import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import subprocess
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#Authentication - without user
client_credentials_manager = SpotifyClientCredentials(client_id='66f0b03c55dc43329ec9a8795e6f9bf5', client_secret='4d4e2232d4de4d1aa3801d05c852a5ee')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# Getting Data from Top 50 Global Songs
playlist_link = "https://open.spotify.com/playlist/37i9dQZEVXbNG2KDcFcKOF?si=1333723a6eff4b7f"
playlist_URI = playlist_link.split("/")[-1].split("?")[0]
track_uris = [x["track"]["uri"] for x in sp.playlist_tracks(playlist_URI)["items"]]

# Initialize empty lists for data
data = []
for track in sp.playlist_tracks(playlist_URI)["items"]:

    # Extract data for each track
    track_uri = track["track"]["uri"]
    track_name = track["track"]["name"]
    artist_uri = track["track"]["artists"][0]["uri"]
    artist_info = sp.artist(artist_uri)
    artist_name = track["track"]["artists"][0]["name"]
    artist_pop = artist_info["popularity"]
    artist_genres = artist_info["genres"]
    album = track["track"]["album"]["name"]
    track_pop = track["track"]["popularity"]


    # Get audio features
    audio_features = sp.audio_features([track_uri])[0]
    acousticness = audio_features["acousticness"]
    danceability = audio_features["danceability"]
    energy = audio_features["energy"]
    instrumentalness = audio_features["instrumentalness"]
    liveness = audio_features["liveness"]
    loudness = audio_features["loudness"]
    speechiness = audio_features["speechiness"]
    tempo = audio_features["tempo"]
    valence = audio_features["valence"]

    # Append the data to the list
    data.append([track_uri, track_name, artist_name, artist_pop, artist_genres, album, track_pop,
                 acousticness, danceability, energy, instrumentalness, liveness, loudness,
                 speechiness, tempo, valence])

from sklearn.preprocessing import MinMaxScaler

# Create a DataFrame
columns = ["Track_URI", "Track_Name", "Artist_Name", "Artist_Popularity", "Artist_Genres", "Album", "Track_Popularity",
           "Acousticness", "Danceability", "Energy", "Instrumentalness", "Liveness", "Loudness",
           "Speechiness", "Tempo", "Valence"]

df = pd.DataFrame(data, columns=columns)

# Use MinMaxScaler to scale the numerical columns
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.iloc[:, 7:])  # Start from the 8th column (Acousticness) to the last column (Valence)

# Replace the scaled data back to the DataFrame
df.iloc[:, 7:] = scaled_data

# run data collection and preprocessing
#subprocess.run(['python','datacollection.py'])

# Create a sample DataFrame with audio features
#df = pd.read_excel('SpotifyTop50Global.xlsx')


#Start of the Page
st.set_page_config(page_title="Audio Analysis", layout="centered")

# Streamlit app
st.markdown('<p style="font-size: 60px; font-weight: bold;">Spotify Top 50 Global Audio Analysis</p>', unsafe_allow_html=True)
st.markdown('<p style="font-weight:bold; font-size: 20px; color: #808080;">By Nizar Mazlan', unsafe_allow_html=True)


# App Detail
with st.expander("App Details"):
    st.write('''
    The Spotify Top 50 Global Audio Analysis assigns varying weightages to different metrics based on their relevance to specific song features, reflecting the author's and song's background.

    Similarity Function is based on K-Means Clustering, to identify similar songs based on audio features.

    Note: Only songs from the Top 50 Global list are included for analysis.
    ''')


# Audio Features
with st.expander("Audio Features"):
    st.write('''
1. Acousticness: A measure of how acoustic a track is. A value of 1.0 indicates high acousticness, while 0.0 indicates low acousticness.

2. Danceability: Describes how suitable a track is for dancing based on a combination of musical elements. Higher values indicate tracks that are more danceable.

3. Energy: Represents a perceptual measure of intensity and activity. Energetic tracks typically feel fast, loud, and noisy.

4. Instrumentalness: Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the more likely the track contains no vocal content.

5. Liveness: Indicates the presence of an audience in the recording. A higher liveness value suggests that the track was likely recorded during a live performance.

6. Loudness: Represents the overall loudness of a track in decibels (dB). Values typically range from -60 to 0 dB.

7. Speechiness: Detects the presence of spoken words in a track. Values above 0.66 are likely to be tracks that are entirely speech, while values below 0.33 suggest non-speech-like tracks.

8. Tempo: Represents the overall estimated tempo of a track in beats per minute (BPM).

9. Valence: Describes the musical positiveness of a track. Tracks with high valence sound more positive, while those with low valence sound more negative.

    ''')

# Explanation of K-Means Clustering
with st.expander("K-Means Clustering"):
    st.write('''
K-Means Clustering is an unsupervised machine learning algorithm that groups data points into a predefined number of clusters (K) based on their similarity. 
It is a simple and efficient algorithm that can be used to identify patterns and relationships in data, and it is widely used in a variety of applications, including data segmentation, image processing, and natural language processing.

How it works:
The K-Means clustering algorithm works by iteratively assigning data points to clusters. The algorithm starts by randomly initializing K centroids, which are the centers of the clusters. Then, each data point is assigned to the cluster with the closest centroid. Once all data points have been assigned to clusters, the centroids are updated to be the average of the data points in the cluster. This process is repeated until the centroids no longer change, which indicates that the algorithm has converged to a final clustering.
             
    ''')


# Select a track from a dropdown in the sidebar
with st.sidebar:
    st.markdown('<h1 style="font-family: Consolas; font-size: 34px;">Select a Song Here...</h1>', unsafe_allow_html=True)
    selected_track = st.selectbox('Select a Song', df['Track_Name'])


# Function to create the radar chart for audio features
def create_radar_chart(track_data):
    del track_data
    fig = px.line_polar(track_data, 
                        r=track_data[['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Valence']].values[0], 
                        theta=['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Valence'], 
                        line_close=True)
    fig.update_traces(fill='toself')
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickvals=np.arange(0, 1.1, 0.2),
                angle=45,
                tickmode='array',
                ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                tickfont=dict(color='black')
            ),
        )
    )
    
    return fig



# Retrieve the selected track's data
selected_track_data = df[df['Track_Name'] == selected_track]

# Display the selected track's name
if not selected_track_data.empty:
    # Compute K-Means Clustering
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=0).fit(df[['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Valence']])
    
    # Get cluster label of the selected track
    selected_track_cluster = kmeans.predict(selected_track_data[['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Valence']])[0]

    # Get indices of tracks in the same cluster
    cluster_indices = np.where(kmeans.labels_ == selected_track_cluster)[0]

    # Compute pairwise distances between the selected track and tracks in the same cluster
    distances = pairwise_distances_argmin_min(selected_track_data[['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Valence']],
                                            df[['Acousticness', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo', 'Valence']].iloc[cluster_indices])[1]

    # Find the index of the most similar track excluding the selected track
    cluster_indices_without_selected = cluster_indices[cluster_indices != selected_track_data.index[0]]
    most_similar_index = cluster_indices_without_selected[np.argmin(distances)]
    most_similar_track = df.iloc[most_similar_index]


    # Create a box with the song information
    # Create the title/header with the specified formatting
    st.markdown(
        f"<h2 style='text-align: center; font-size: 32px; margin: 0;'>{selected_track_data['Track_Name'].values[0]}</h2>"
        f"<p style='text-align: center; font-size: 24px; margin: 0;'>{selected_track_data['Artist_Name'].values[0]}</p>"
        f"<p style='text-align: center; font-size: 18px; margin: 0;'>Album: {selected_track_data['Album'].values[0]}</p>"
        f"<p style='text-align: center; font-size: 14px; margin: 0;'>Popularity: {selected_track_data['Track_Popularity'].values[0]}</p>",
        unsafe_allow_html=True
    )

    # Display the radar chart for audio features
    st.plotly_chart(create_radar_chart(selected_track_data), use_container_width=True)

    try:
        genre_string = selected_track_data["Artist_Genres"].values[0]
        genres_list = json.loads(genre_string)
    except json.JSONDecodeError:
        # If the JSON string is invalid, set the genre list to an empty list
        genres_list = []
    # Join the list of genres
    genres = ', '.join(genres_list)
    # Display a success message and a warning message
    st.success(f'**Artist Genres:** {genres}')
    st.warning(f'**Artist Popularity:** {selected_track_data["Artist_Popularity"].values[0]}')
    
    # Display the most similar track information in a styled box
    st.markdown(
        """
        <div style="justify-content: center; align-items: center;background-color: #808080; padding: 15px; border-radius: 10px;">
            <h3 style="text-align: center;">Most Similar Track</h3>
            <p style="text-align: center;"><strong>Track Name:</strong> {}</p>
            <p style="text-align: center;"><strong>Artist:</strong> {}</p>
            <p style="text-align: center;"><strong>Album:</strong> {}</p>
            <p style="text-align: center;"><strong>Popularity:</strong> {}</p>
        </div>
        """.format(
            most_similar_track['Track_Name'],
            most_similar_track['Artist_Name'],
            most_similar_track['Album'],
            most_similar_track['Artist_Popularity']
        ),
        unsafe_allow_html=True
    )



