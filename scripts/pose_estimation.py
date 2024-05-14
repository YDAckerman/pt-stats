from ultralytics import YOLO

# #############################
# basic pose estimation example
# #############################

model = YOLO('/home/yoni/Projects/learn-stats/pt-stats/models/yolov8m-pose.pt')

vid_path = '/home/yoni/Projects/learn-stats/pt-stats/raw_video/animation_reference' + \
    '/Athletic Male Standard Walk Animation Reference Body Mechanics.mp4'

results = model(source=vid_path, show=True, conf=0.3, show_labels=True)
