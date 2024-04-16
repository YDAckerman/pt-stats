from ultralytics import YOLO
# from src.pose import ResultHandler as rh
import pickle

model = YOLO('/home/yoni/Projects/learn-stats/pt-stats/models/yolov8m-pose.pt')

results = model(source='/home/yoni/Videos/Webcam/2024-04-15-100300.webm',
                show=False, conf=0.3, show_labels=True)

with open('results_4_15_24.pkl', 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)


# handler = rh(results)

# outputs are saved in runs/post/predict as avi files
# these can be watched later with 'xdg-open [file name]'
# results = model(source='/home/yoni/Videos/vlc-record-' +
#                 '2024-04-01-12h11m15s-v4l2____dev_video0-.avi',
#                 show=True, conf=0.3, save=True, save_txt=True)

# results = model(source='/home/yoni/Videos/vlc-record-' +
#                 '2024-04-01-12h11m15s-v4l2____dev_video0-.avi',
#                 show=True, conf=0.3, show_labels=True)

# results = model(source='https://youtu.be/GBkJY86tZRE',
#                 show=True, conf=0.3, show_labels=True)

# distance calc
# pt_dist = ((pt_f2[0] - pt_f1[0])**2 + (pt_f2[1] - pt_f1[1])**2)**0.5
    
# results = model(source=0,
#                 show=True, conf=0.3,
#                 show_labels=True)
# results = model(source=0, show=True, conf=0.3, save=True)

# results[0].keypoints roughly corresponds to the labels output in runs/pose/predict/labels
# but I haven't quite figured that all out yet.

