import os

from pipeline.landmarks import predict_bone_joint_model


class BoneJoints(object):

    def predict(self, subject_directory, n_images=35):
        os.chdir(subject_directory)
        predict_bone_joint_model(n_images)
