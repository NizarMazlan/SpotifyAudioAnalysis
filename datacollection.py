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

import pandas as pd
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

# Save it to an excel file
df.to_excel("SpotifyTop50Global.xlsx")