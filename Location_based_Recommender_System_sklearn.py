import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_json('restaurant.json', lines=True)
df.head()
df.shape
df['Restaurants'] = df['categories'].str.contains('Restaurants')
df.head(2)
df_res = df.loc[df.Restaurants == True]
df_res.head()
df_res.shape

fig, ax = plt.subplots(figsize=(12, 10))
sns.countplot(df_res['stars'], ax=ax)
plt.title('Star Plot')
plt.show()
top_res = df_res.sort_values(
    by=['review_count', 'stars'], ascending=False)[:20]
top_res.head()
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x='stars', y='name', data=top_res, ax=ax)
plt.savefig('top20_res.png')
plt.show()

px.scatter_mapbox(df_res, latitude='lat', longtitude='long', color='stars',
                  size='review_count', size_max=30, zoom=3, width=1200, height=800)
lasVegas = df_res[df_res.state == 'NV']
px.scatter_mapbox(lasVegas, latitude='lat', longtitude='long', color='stars',
                  size='review_count', size_max=15, zoom=10, width=1200, height=800)
coords = lasVegas[['long', 'lat']]
distortions = []

K = range(1, 25)
for k in K:
    kmeansModel = KMeans(n_clusters=k)
    kmeansModel = kmeansModel.fit(coords)
    distortions.append(kmeansModel.inertia_)
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(K, distortions, marker='o')
plt.xlabel('k')
plt.ylabel('Distortions')
plt.title('Elbow Method For Optimal k')
plt.show()

sil = []
kmax = 50
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters=k).fit(coords)
    labels = kmeans.labels_
    sil.append(silhouette_score(coords, labels, metric='euclidean'))
sil
kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans.fit(coords)
y = kmeans.labels_
print('k = 5', ' silhouette_score ',
      silhouette_score(coords, y, metric='euclidean'))
lasVegas['cluster'] = kmeans.predict(lasVegas[['long', 'lat']])
lasVegas.head()

px.scatter_mapbox(lasVegas, latitude='lat', longtitude='long', color='cluster', size='review_count',
                  hover_data=['name', 'lat', 'long'], zoom=10, width=1200, height=800)
top_res = lasVegas.sort_values(
    by=['review_count', 'stars'], ascending=False)
top_res.head()


def recommend_res(df, long, lat):
    cluster = kmeans.predict(np.array([long, lat]).reshape(1, -1))[0]
    print(cluster)
    return df[df['cluster'] == cluster].iloc[0:5][['name', 'lat', 'long']]


recommend_res(top_res, -115.1891691, 36.1017316)
recommend_res(top_res, -115.2798544, 36.0842838)
recommend_res(top_res, -115.082821, 36.155011)
test_coordinates = {'user': [1, 2, 3], 'lat': [
    36.1017316, 36.0842838, 36.155011], 'long': [-115.1891691, -115.2798544, -115.082821]}
test_df = pd.DataFrame(test_coordinates)
test_df
user1 = test_df[test_df['user'] == 1]
user1

fig = px.scatter_mapbox(recommend_res(top_res, user1.long, user1.lat),
                        latitude='lat', longtitude='long', zoom=10, width=1200, height=800, hover_data=['name', 'lat', 'long'])
fig.add_scattermapbox(latitude=user1['lat'], longtitude=user1['long']).update_traces(
    dict(mode='markers', marker=dict(size=15)))
user2 = test_df[test_df['user'] == 2].reset_index()
fig = px.scatter_mapbox(recommend_res(top_res, user2.long, user2.lat),
                        latitude='lat', longtitude='long', zoom=10, width=1200, height=800, hover_data=['name', 'lat', 'long'])
fig.add_scattermapbox(latitude=user2['lat'], longtitude=user2['long']).update_traces(
    dict(mode='markers', marker=dict(size=15)))
user3 = test_df[test_df['user'] == 2].reset_index()
fig = px.scatter_mapbox(recommend_res(top_res, user3.long, user3.lat),
                        latitude='lat', longtitude='long', zoom=10, width=1200, height=800, hover_data=['name', 'lat', 'long'])
fig.add_scattermapbox(latitude=user3['lat'], longtitude=user3['long']).update_traces(
    dict(mode='markers', marker=dict(size=15)))
