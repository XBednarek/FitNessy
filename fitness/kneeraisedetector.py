# Import relatif dans le package :
from .movedetector import MoveDetector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calc_distance
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.kneeraisedetector

# Autres imports
import numpy as np
import cv2

class KneeRaiseDetector(MoveDetector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False,
                                             show_landmark:bool = False,
                                             windows_name:str = cst.WIN_NAME_KNEERAISE) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose=verbose,
                                               show_landmark=show_landmark,
                                               windows_name=windows_name,
                                               reward_string = "knee raises !",
                                               movement_cycle = ["left_up_right_down", "left_down_right_up"],
                                               exo_name = "Knee raises")

    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    # Implémentation de la méthode abstraite de la classe mère Detector
    def detect(self, positions: np.ndarray) -> str:
        """Détecte une position"""
        if self.kneeraise_left_up_detector(positions) and self.kneeraise_right_down_detector(positions):
            return 'left_up_right_down'
        elif self.kneeraise_left_down_detector(positions) and self.kneeraise_right_up_detector(positions):
            return 'left_down_right_up'
        else :
            return 'other'
        
    # -------------------------------------------------------------------------
    #                                                        Méthodes statiques
    # -------------------------------------------------------------------------
    @staticmethod
    def kneeraise_left_up_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the left up pose.
        """
        # if landmarks:
        left_knee_shoulder_dist = calc_distance(landmarks, 26, 12)
        left_hip_shoulder_dist = calc_distance(landmarks, 24, 12)
        if left_knee_shoulder_dist < left_hip_shoulder_dist * 1.3:
            return True

        return False


    @staticmethod
    def kneeraise_right_up_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the right up pose.
        """
        # if landmarks:
        right_knee_shoulder_dist = calc_distance(landmarks, 25, 11)
        right_hip_shoulder_dist = calc_distance(landmarks, 23, 11)
        if right_knee_shoulder_dist < right_hip_shoulder_dist * 1.3:
            return True

        return False

    @staticmethod
    def kneeraise_left_down_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the left down pose.
        """
        # if landmarks:
        left_knee_shoulder_dist = calc_distance(landmarks, 26, 12)
        left_hip_shoulder_dist = calc_distance(landmarks, 24, 12)
        if left_knee_shoulder_dist > left_hip_shoulder_dist * 1.5:
            return True

        return False

    @staticmethod
    def kneeraise_right_down_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the right down pose.
        """
        # if landmarks:
        right_knee_shoulder_dist = calc_distance(landmarks, 25, 11)
        right_hip_shoulder_dist = calc_distance(landmarks, 23, 11)
        if right_knee_shoulder_dist > right_hip_shoulder_dist * 1.5:
            return True

        return False

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe KneeRaiseDetector *")
    print("-"*80)
    import mediapipe as mp
    # Modèle mediapipe :
    mediapipe_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Capture vidéo :
    cap = cv2.VideoCapture(0)
    # Réglage de la verbosité
    verbose = True
    kneeraise_detector = KneeRaiseDetector(mediapipe_model, cap, verbose)
    kneeraise_detector.run(objective=4)