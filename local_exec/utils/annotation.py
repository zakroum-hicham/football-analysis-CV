import supervision as sv
from .graphics import draw_team_ball_control
colors = {
    "team1":sv.ColorPalette.from_hex(['#1E90FF']),
    "team2":sv.ColorPalette.from_hex(['#DC143C']),
    "referee":sv.ColorPalette.from_hex(['#FFD700']),
    "goalkepper":sv.ColorPalette.from_hex(['#FFF']),
    "label_text":sv.Color.from_hex('#000'),
    "ball":sv.Color.from_hex('#FF8C00'),
    "active_player":sv.Color.from_rgb_tuple((255,0,0))
}

LABEL_TEXT_POSITION = sv.Position.BOTTOM_CENTER


team1_ellipse_annotator = sv.EllipseAnnotator(color=colors["team1"])  
team2_ellipse_annotator = sv.EllipseAnnotator(color=colors['team2'])
referee_ellipse_annotator = sv.EllipseAnnotator(color=colors['referee'])
goalkepper_ellipse_annotator = sv.EllipseAnnotator(color=colors['goalkepper'])
active_player_annotator = sv.TriangleAnnotator(color=colors['active_player'],base=18,height=18,outline_color=sv.Color.BLACK,outline_thickness=1)

team1_label_annotator = sv.LabelAnnotator(color=colors['team1'],text_color=colors['label_text'], text_position=LABEL_TEXT_POSITION)
team2_label_annotator = sv.LabelAnnotator(color=colors['team2'],text_color=colors['label_text'],text_position=LABEL_TEXT_POSITION)
referee_label_annotator = sv.LabelAnnotator(color=colors['referee'],text_color=colors['label_text'],border_radius=30, text_position=LABEL_TEXT_POSITION)
goalkepper_label_annotator = sv.LabelAnnotator(color=colors['goalkepper'],text_color=colors['label_text'],border_radius=30,text_position=LABEL_TEXT_POSITION)

ball_triangle_annotator = sv.TriangleAnnotator(color=colors['ball'], base=18, height=18,outline_color=sv.Color.BLACK,outline_thickness=1)


# annotate the frames
def annotate_frames(frame,all_detection,labels,ball_posession):
    annotated_frame = frame.copy()
    annotated_frame = team1_ellipse_annotator.annotate(scene=annotated_frame, detections=all_detection["team1"])
    annotated_frame = team2_ellipse_annotator.annotate(scene=annotated_frame, detections=all_detection["team2"])
    annotated_frame = referee_ellipse_annotator.annotate(scene=annotated_frame, detections=all_detection["referee"])
    annotated_frame = goalkepper_ellipse_annotator.annotate(scene=annotated_frame,detections=all_detection["goalkeepers"])
    annotated_frame = ball_triangle_annotator.annotate(scene=annotated_frame, detections=all_detection["ball"])
    annotated_frame = team1_label_annotator.annotate(scene=annotated_frame, detections=all_detection["team1"], labels=labels["labels_team1"])
    annotated_frame = team2_label_annotator.annotate(scene=annotated_frame, detections=all_detection["team2"], labels=labels["labels_team2"])
    annotated_frame = referee_label_annotator.annotate(scene=annotated_frame,detections=all_detection["referee"],labels=labels["labels_referee"])
    annotated_frame = goalkepper_label_annotator.annotate(scene=annotated_frame,detections=all_detection["goalkeepers"],labels=labels["labels_gk"])
    annotated_frame = active_player_annotator.annotate(scene=annotated_frame,detections=all_detection["active_player"]) 
    annotated_frame = draw_team_ball_control(annotated_frame,ball_posession)
    return annotated_frame