# Import relatif dans le package :
from .posedetector import PoseDetector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calc_distance
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.treeposedetector

# Autres imports
import numpy as np
import cv2
from queue import Queue

class TreePoseDetector(PoseDetector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False,
                                             show_landmark:bool = False,
                                             windows_name:str = cst.WIN_NAME_HEELS2BUTTOCKS,
                                             fast_api_queues:tuple[Queue, Queue]|None = None) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose=verbose,
                                               show_landmark=show_landmark,
                                               windows_name=windows_name,
                                               fast_api_queues=fast_api_queues,
                                               reward_string = "sec. of tree pose !",
                                               pose_to_keep = "tree_pose",
                                               exo_name = "Tree Pose")
        
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    # Implémentation de la méthode abstraite de la classe mère Detector
    def detect(self, positions: np.ndarray, visibility: np.ndarray) -> str:
        """Détecte une position"""
        if self.treepose_detector(positions):
            return 'tree_pose'
        else :
            return 'other'

    # -------------------------------------------------------------------------
    #                                                        Méthodes statiques
    # -------------------------------------------------------------------------
    @staticmethod
    def treepose_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the tree pose.
        """
        # if landmarks:
        hip_distance = calc_distance(landmarks, 24, 23)
        left_wrist_right_wrist_dist = calc_distance(landmarks, 15, 16)
        left_knee_right_knee_dist = calc_distance(landmarks, 26, 25)
        # detecting left tree pose
        left_ankle_hip_dist = calc_distance(landmarks, 27, 23)
        right_ankle_left_knee_dist = calc_distance(landmarks, 28, 25)
        right_foot_left_knee_dist = calc_distance(landmarks, 32, 25)
        # detecting right tree pose
        right_ankle_hip_dist = calc_distance(landmarks, 28, 24)
        left_ankle_right_knee_dist = calc_distance(landmarks, 27, 26)
        left_foot_right_knee_dist = calc_distance(landmarks, 31, 26)

        if (left_ankle_hip_dist > 1.5*hip_distance
            and (right_ankle_left_knee_dist < hip_distance or right_foot_left_knee_dist < hip_distance)
            and left_wrist_right_wrist_dist < hip_distance
            and left_knee_right_knee_dist > hip_distance
            ):
            return True
        elif (right_ankle_hip_dist > 1.5*hip_distance
            and (left_ankle_right_knee_dist < hip_distance or left_foot_right_knee_dist < hip_distance)
            and left_wrist_right_wrist_dist < hip_distance
            and left_knee_right_knee_dist > hip_distance
            ):
            return True

        return False

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe TreePoseDetector *")
    print("-"*80)
    import mediapipe as mp
    # Modèle mediapipe :
    mediapipe_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Capture vidéo :
    cap = cv2.VideoCapture(0)
    # Réglage de la verbosité
    verbose = True
    treepose_detector = TreePoseDetector(mediapipe_model, cap, verbose, show_landmark=True)
    treepose_detector.run(objective=4)
