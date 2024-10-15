import numpy as np


MAX_DISTANCE = 70
    

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def measure_distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def assign_ball_to_player(players_detections,ball_bbox):
    if len(ball_bbox)==0 or len(players_detections.xyxy) == 0:
        return -1
    ball_position = get_center_of_bbox(ball_bbox[0])

    miniumum_distance = 99999
    assigned_player=-1

    for object_ind , _ in enumerate(players_detections.class_id):
        player_bbox = players_detections.xyxy[object_ind]

        distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
        distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
        distance = min(distance_left,distance_right)

        if distance < MAX_DISTANCE:
            if distance < miniumum_distance:
                miniumum_distance = distance
                assigned_player = object_ind

    return assigned_player