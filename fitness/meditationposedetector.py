# Import relatif dans le package :
from .posedetector import PoseDetector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calc_distance
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.meditationposedetector

# Autres imports
import numpy as np
import cv2
from queue import Queue

class MeditationPoseDetector(PoseDetector):

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
                                               reward_string = "sec. of meditation pose !",
                                               pose_to_keep = "meditation_pose",
                                               exo_name = "Meditation Pose")
        
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    # Implémentation de la méthode abstraite de la classe mère Detector
    def detect(self, positions: np.ndarray, visibility: np.ndarray) -> str:
        """Détecte une position"""
        if self.meditationpose_detector(positions):
            return 'meditation_pose'
        else :
            return 'other'

    # -------------------------------------------------------------------------
    #                                                        Méthodes statiques
    # -------------------------------------------------------------------------
    @staticmethod
    def meditationpose_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the meditation pose.
        """
        # if landmarks:
        hip_distance = calc_distance(landmarks, 24, 23)
        left_wrist_right_wrist_dist = calc_distance(landmarks, 15, 16)
        left_knee_right_knee_dist = calc_distance(landmarks, 26, 25)
        left_wrist_knee_dist = calc_distance(landmarks, 15, 25)
        right_wrist_knee_dist = calc_distance(landmarks, 16, 26)
        left_pinky_knee_dist = calc_distance(landmarks, 17, 25)
        right_pinky_knee_dist = calc_distance(landmarks, 18, 26)
        left_thumb_knee_dist = calc_distance(landmarks, 21, 25)
        right_thumb_knee_dist = calc_distance(landmarks, 22, 26)
        left_index_knee_dist = calc_distance(landmarks, 19, 25)
        right_index_knee_dist = calc_distance(landmarks, 20, 26)
        
        left_foot_right_hip_distance = calc_distance(landmarks, 31, 24)
        left_foot_left_hip_distance = calc_distance(landmarks, 31, 23)
        right_foot_left_hip_distance = calc_distance(landmarks, 32, 23)
        right_foot_right_hip_distance = calc_distance(landmarks, 32, 24)

        if (left_wrist_right_wrist_dist > hip_distance
            and left_knee_right_knee_dist > hip_distance
            and (left_wrist_knee_dist < hip_distance
                 or left_pinky_knee_dist < hip_distance
                 or left_thumb_knee_dist < hip_distance
                 or left_index_knee_dist < hip_distance)
            and (right_wrist_knee_dist < hip_distance
                 or right_pinky_knee_dist < hip_distance
                 or right_thumb_knee_dist < hip_distance
                 or right_index_knee_dist < hip_distance)
            and left_foot_right_hip_distance < right_foot_right_hip_distance
            and right_foot_left_hip_distance < left_foot_left_hip_distance
            ):
            return True

        return False

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe MeditationPoseDetector *")
    print("-"*80)
    import mediapipe as mp
    # Modèle mediapipe :
    mediapipe_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Capture vidéo :
    cap = cv2.VideoCapture(0)
    # Réglage de la verbosité
    verbose = True
    meditationpose_detector = MeditationPoseDetector(mediapipe_model, cap, verbose, show_landmark=True)
    meditationpose_detector.run(objective=4)
