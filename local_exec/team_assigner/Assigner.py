from sklearn.cluster import KMeans
import cv2
class Assigner:
    def __init__(self) -> None:
         self.team_colors={}

    def get_clustering_model(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_2d = image.reshape(-1,3)
        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2,n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_team(self,frame, player_bbox,kmeans):
            player_color = self.get_player_color(frame, player_bbox)
            team_id = kmeans.predict(player_color.reshape(1, -1))[0]
            return team_id

    def get_player_color(self,frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self,frame, players_detections):
            
            player_colors = []
            for xy2 in players_detections.xyxy:
                bbox = xy2
                player_color = self.get_player_color(frame, bbox)
                player_colors.append(player_color)
            
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto")
            kmeans.fit(player_colors)
            self.team_colors[1] = kmeans.cluster_centers_[0]
            self.team_colors[2] = kmeans.cluster_centers_[1]
            return kmeans