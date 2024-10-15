import cv2
import supervision as sv
from config import *
def draw_team_ball_control(frame,ball_control):
        
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1210, 145), (1838,70), (255,255,255), -1 )
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_1 = ball_control[MODEL_CLASSES["team1"]] / (ball_control[MODEL_CLASSES["team1"]]+ball_control[MODEL_CLASSES["team2"]])
        team_2 = ball_control[MODEL_CLASSES["team2"]] / (ball_control[MODEL_CLASSES["team1"]]+ball_control[MODEL_CLASSES["team2"]])

        cv2.putText(frame,"Ball Control : ",(1220,115),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,0),3)
        cv2.putText(frame, f"{team_1*100:.2f}%",(1500,115), cv2.FONT_HERSHEY_SIMPLEX, 1, sv.Color.as_bgr(sv.Color.from_hex('#1E90FF')), 3)
        cv2.putText(frame, f"{team_2*100:.2f}%",(1700,115), cv2.FONT_HERSHEY_SIMPLEX, 1, sv.Color.as_bgr(sv.Color.from_hex('#DC143C')), 3)

        return frame