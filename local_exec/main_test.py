from ultralytics import YOLO
import supervision as sv
import cv2
from tqdm import tqdm
from utils import get_number_of_frames , annotate_frames,assign_ball_to_player,get_frames
from team_assigner import Assigner
import numpy as np
from config import *


def main():
    model = YOLO(MODEL_SRC)
    
    # get the total frames
    total_frames , fps = get_number_of_frames(VIDEO_SRC)

    # get the frames
    frame_generator = get_frames(VIDEO_SRC)

    # Get the first frame to determine the video dimensions
    first_frame = next(frame_generator)
    height, width, _ = first_frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))
    
    # Initialize the ByteTrack object
    tracker = sv.ByteTrack()
    tracker.reset()
    tracker1 = sv.ByteTrack()
    tracker1.reset()


    # Process each frame in the video
    team_colors = {}
    is_first_frame = True
    class_id_active_player = None
    kmeans = None
    ball_posession = {MODEL_CLASSES["team1"]:0,MODEL_CLASSES["team2"]:0}
    ########################## for debuging
    i = 1
    ##########################

    for frame in tqdm(frame_generator,total=total_frames):
            
            if (i%10 != 0) or i<110:
                i+=1
                continue
            i+=1
            
        # try:
            # Predict the boxes with the model
            result = model.predict(frame, conf=0.3,verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # saperate the detections
            goalkeepers_detections = detections[detections.class_id == MODEL_CLASSES["goalkepper"]]
            

             # convert goalkeeper to player
            # for object_ind , class_id in enumerate(detections.class_id):
            #     if class_id == MODEL_CLASSES["goalkepper"]:
            #         detections.class_id[object_ind] = MODEL_CLASSES["player"]

                
            ball_detections = detections[detections.class_id == MODEL_CLASSES["ball"]]
            players_detections = detections[detections.class_id == MODEL_CLASSES["player"]]
            referee_detections = detections[detections.class_id == MODEL_CLASSES["referee"]]




            # Appling Non-Maximum Suppression (NMS) to the players detections
            players_detections = players_detections.with_nms(threshold=0.5)
            
            
            
            # Assign Player Teams
            if is_first_frame:
                team_assigner = Assigner()
                kmeans = team_assigner.assign_team_color(frame, players_detections)
                team_colors = team_assigner.team_colors
                is_first_frame = False
            for object_ind , _ in enumerate(players_detections.class_id):
                player_color = team_assigner.get_player_color(frame,players_detections.xyxy[object_ind])
                team_id = team_assigner.get_player_team(frame, players_detections.xyxy[object_ind],kmeans)
                if team_id == 0:
                    players_detections.class_id[object_ind] = MODEL_CLASSES["team1"]
                    a = np.sqrt((player_color[0] - team_colors[1][0])**2 + (player_color[1] - team_colors[1][1])**2 + (player_color[2]- team_colors[1][2])**2)
                    if a > 110 and a < 180:
                        players_detections.class_id[object_ind] = MODEL_CLASSES["goalkepper"]
                elif team_id == 1:
                    players_detections.class_id[object_ind] = MODEL_CLASSES["team2"]
                    a = np.sqrt((player_color[0] - team_colors[2][0])**2 + (player_color[1] - team_colors[2][1])**2 + (player_color[2] - team_colors[2][2])**2)
                    if a > 110 and a < 180:
                        players_detections.class_id[object_ind] = MODEL_CLASSES["goalkepper"]
            
            
            goalkeepers_detections1 = players_detections[players_detections.class_id == MODEL_CLASSES["goalkepper"]]
            goalkeepers_detections = sv.Detections.merge([goalkeepers_detections,goalkeepers_detections1])
            
            team1_detections =  players_detections[players_detections.class_id == MODEL_CLASSES["team1"] ]
            team2_detections =  players_detections[players_detections.class_id == MODEL_CLASSES["team2"] ]
           
            
            players_detections = players_detections[players_detections.class_id != MODEL_CLASSES["goalkepper"]]
            # assign the ball to the closest player
            player_ind = assign_ball_to_player(players_detections,ball_detections.xyxy)
            all_players = players_detections
            if player_ind != -1:
                class_id_active_player = players_detections.class_id[player_ind]
                ball_posession[class_id_active_player]+=1
                all_players.class_id[player_ind] = MODEL_CLASSES["active_player"]
            elif class_id_active_player:
                ball_posession[class_id_active_player]+=1
            active_player_detection = all_players[all_players.class_id == MODEL_CLASSES["active_player"]]

            #  adding a padding to the ball detection and active player
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            active_player_detection.xyxy = sv.pad_boxes(xyxy=active_player_detection.xyxy, px=10)

            # add tracker teams
            team1_detections = tracker.update_with_detections(detections=team1_detections)
            team2_detections = tracker1.update_with_detections(detections=team2_detections)

            # creating labels 

            labels = {
                "labels_team1" :[f"{tracker_id}" for tracker_id in team1_detections.tracker_id],
                "labels_team2" : [f"{tracker_id}" for tracker_id in team2_detections.tracker_id],
                "labels_referee" : ["ref"] * len(referee_detections),
                "labels_gk" : ["GK"] * len(goalkeepers_detections)
                 
            }
            
            # Annotate the frame

            all_detection = {
                 "goalkeepers":goalkeepers_detections,
                 "ball":ball_detections,
                 "palyers":players_detections,
                 "referee":referee_detections,
                 "team1":team1_detections,
                 "team2":team2_detections,
                 "active_player":active_player_detection
            }
            annotated_frame = annotate_frames(frame,all_detection,labels,ball_posession)
            # Write the annotated frame to the output video
            # out.write(annotated_frame)
            sv.plot_image(annotated_frame)
            # print(ball_posession)
    #     except Exception as e:
    #         print(f"Error processing frame: {e}")
    #         continue
            # break
    # Release resources
    out.release()
    # cv2.destroyAllWindows()

    print(f"Video saved as {OUT_VIDEO}")

if __name__ == '__main__':
    main()