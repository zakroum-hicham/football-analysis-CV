# --> Football Analysis Project
##  Project Overview
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, a leading AI object detection model. I also trained the model to improve its performance. Additionally, we assigned players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. We also assigned the ball to a player during gameplay. With this information, we measured a team's ball acquisition percentage throughout the match. Furthermore, we implemented perspective transformation  to convert the scene into a 2D graphic,, allowing us to measure a player's movement in meters rather than pixels.
<br />
<br />

<p align="center">
  <img src="https://github.com/user-attachments/assets/c8feea88-035f-44eb-8a90-b68308426557">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1282ec03-f1ff-4829-bf6f-0fc44665f933">
</p>

## Features
- Object Detection: Detect and track players, referees, and footballs using YOLO.
- Player-Team Assignment: Assign players to teams based on t-shirt colors using KMeans clustering.
- Ball Assignment: Assign the ball to a player during gameplay.
- Ball Possession Analysis: Measure team ball possession throughout the match.
- Movement Analysis: Track player movements and measure distances in meters using perspective transformation.

## Tech Stack

- Programming Language: Python
- Framework: TensorFlow/PyTorch (for YOLO)
- Libraries: OpenCV, NumPy, scikit-learn, matplotlib, supervision, ultralytics
- Object Detection Model: YOLO (You Only Look Once)
- Clustering Algorithm: KMeans (for team assignment)
- Transformation: Perspective transformation (for real-world measurement)

## Trained YOLO Model Results
  ### Model Summary (Training)
  ![training_res](https://github.com/user-attachments/assets/91d8b3aa-d453-41f5-89b9-4a29849bde57)
  ### Model Summary (Validation)
  ![validation_res](https://github.com/user-attachments/assets/4a4fc3f4-71f7-40f1-ac89-7c9c270a3a3d)
  ### Result
  ![results](https://github.com/user-attachments/assets/1b17aaa4-116a-40d3-9f31-3d540281aa28)
## Final Result (Demo) 


https://github.com/user-attachments/assets/7f43e4f1-dea5-4c83-bf41-fa462d4992b4


## Challenges and Improvements
  
  - Challenges:
    
    * Distinguishing players with similar t-shirt colors.
    * Tracking Players in complex game scenarios.


  - Potential Improvements:
 
    * Training the YOLO with more Data (I only use a small DataSet)
    * Improve goalkepper team asigner
    * Use more advanced clustering techniques for better player-team assignment.
