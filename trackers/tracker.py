from ultralytics import YOLO
import supervision as sv

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
            break
        return detections

    # Tracking
    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks={
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

            print(detection_supervision)

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


            print(detection_supervision)