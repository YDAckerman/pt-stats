import pandas as pd


class ResultHandler():

    labels = [
        "nose", "left_eye",
        "right_eye", "left_ear",
        "right_ear", "left_shoulder",
        "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist",
        "right_wrist", "left_hip",
        "right_hip", "left_knee",
        "right_knee", "left_ankle",
        "right_ankle"
    ]

    def __init__(self, yolo_res):

        self.kpts = [frame.keypoints for frame in yolo_res]
        self.kp_to_df()

    def kp_to_df(self):

        individuals = []

        for i, kpt in enumerate(self.kpts):
            indv_pose = []
            for pose in kpt.xyn:
                pose_df = pd.DataFrame(pose.numpy(),
                                       columns=['X', 'Y'])
                pose_df['label'] = self.labels
                pose_df['frame'] = [i]*len(self.labels)
                indv_pose.append(pose_df)

            individuals.append(indv_pose)

        individuals = zip(*individuals)
        return [pd.concat(indv) for indv in individuals]
