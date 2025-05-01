from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    # model assignment
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        # avoid memory issues
        # divide frames into batches of 20
        batch_size = 20

        detections = []
        # loop over each batch
        for i in range(0, len(frames), batch_size):
            # detections for each batch
            # conf: confidence threshold
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    # Tracking
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # reading
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # if it exists, open up the stub file and load the tracks and return
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks={
            # [{0:{"bbox":[0,0,0,0]]}, {1:{"bbox":[0,0,0,0]]}, {2:{"bbox":[0,0,0,0]]}},
            # {{10:{"bbox":[0,0,0,0]]},{21:{"bbox":[0,0,0,0]]},{1:{"bbox":[0,0,0,0]]}}]
            "players":[], 
            "referees":[],
            "ball":[],
        }

        # Override goalkeeper with player
        # Going over detections
        for frame_num, detection in enumerate(detections):
            # {0:person, 1:goalkeeper, ...}
            cls_names = detection.names
            # inverse {person: 0, goalkeeper:1, ...}
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    # convert to player object
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)    
           
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Loop over each detection for players and referees
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            # Loop over each detection for balls
            for frame_detections in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        # save
        if stub_path is not None:
            # wb: write bites
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)  

        return tracks

    # Drawing ellipse
    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35*width)), angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)
        return frame

    # Making annotations
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                # Draw ellipse
                frame = self.draw_ellipse(frame, player["bbox"], (100, 200, 170), track_id)
                
            output_video_frames.append(frame)
        return output_video_frames