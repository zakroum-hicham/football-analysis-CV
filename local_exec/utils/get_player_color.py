from rembg import remove
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans



def get_player_color(frame,bbox):
    image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    # input_image = Image.open("testing3.png")
    output_image = remove(image)

    output_array = np.asarray(output_image)

    pixels = output_array.reshape(-1, 4) 
    pixels = pixels[pixels[:, 3] != 0]

    # 3. Run KMeans clustering to find multiple dominant colors
    n_clusters = 3  # Set this to a higher number if you expect multiple dominant colors
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels[:, :3])  # Only consider RGB channels, ignore alpha

    # 4. Get the cluster centers (i.e., dominant colors)
    cluster_centers = kmeans.cluster_centers_

    # 5. Find the number of pixels in each cluster (i.e., the size of each cluster)
    _, counts = np.unique(kmeans.labels_, return_counts=True)

    # 6. Sort clusters by size and find the second largest (second most frequent)
    sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order of size
    dominant_color = cluster_centers[sorted_indices[0]]  # Most dominant color
    second_dominant_color = cluster_centers[sorted_indices[1]]  # Second dominant color

    return second_dominant_color