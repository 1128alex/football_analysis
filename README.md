# Football Analysis — Portfolio (Python)

A comprehensive football match analysis toolkit built with Python. This project features player tracking, ball assignment, team classification, camera movement estimation, and speed/distance analytics using computer vision and deep learning (YOLOv5). It is designed for research, sports analytics, and video-based football insights.

- Demo Video: [Sample Output](output_videos/output_video.avi)
- Repository: https://github.com/1128alex/football_analysis
 - Live Web App: https://football-analysis-front.vercel.app/

## ✨ Features
- Player and ball detection using YOLOv5
- Player tracking and ball assignment
- Team classification
- Camera movement estimation
- Speed and distance analytics
- Modular design for easy extension
- Input/output video processing
 - Real-time video upload and analysis via Next.js web app
 - Scalable backend services on AWS EC2 with SQS

## 🛠️ Tech Stack
- Python
- OpenCV
- PyTorch
- YOLOv5
- NumPy, Pandas
- Custom modules for tracking, assignment, analytics
 - Next.js
 - AWS EC2 and SQS
 - Vercel

## 📁 Project Structure (key parts)
```
main.py
requirements.txt
input_videos/
output_videos/
models/
camera_movement_estimator/   # Camera movement estimation
player_ball_assigner/        # Player-ball assignment logic
speed_distance_estimator/    # Speed & distance analytics
team_assigner/               # Team classification
trackers/                    # Player tracking
utils/                       # Utility functions (bbox, video)
view_transformer/            # Perspective transforms
training/
```

## 🚀 Getting Started (Local)

1) Install dependencies
```powershell
pip install -r requirements.txt
```

2) Run the main analysis pipeline
```powershell
python main.py
```

## 🧱 Build/Train
- To retrain YOLOv5: see `training/football_training_yolo_v5.ipynb`
- Place trained weights in `models/`

## 🌐 Cloud & Web Deployment
- Backend services run on AWS EC2, with jobs orchestrated via AWS SQS for scalable video processing.
- Frontend web app for video upload and real-time analysis: https://football-analysis-front.vercel.app/

## 🌐 Usage Notes
- Input videos: place in `input_videos/`
- Output videos/results: saved to `output_videos/` and `runs/`
- Models: use `models/best.pt` or your own YOLO models

## 📄 License
Please credit the author when you use parts of my repository.
