# Import relatif dans le package :
from .movedetector import MoveDetector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calc_distance
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.heels2buttocksdetector

# Autres imports
import numpy as np
import cv2

class Heels2ButtocksDetector(MoveDetector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False,
                                             show_landmark:bool = False,
                                             windows_name:str = cst.WIN_NAME_HEELS2BUTTOCKS) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose=verbose,
                                               show_landmark=show_landmark,
                                               windows_name=windows_name,
                                               reward_string = "heels to buttocks !",
                                               movement_cycle = ["left_up_right_down", "left_down_right_up"],
                                               exo_name = "Heels to buttocks")

    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    # Implémentation de la méthode abstraite de la classe mère Detector
    def detect(self, positions: np.ndarray, visibility: np.ndarray) -> str:
        """Détecte une position"""
        if self.heels2buttocks_left_up_detector(positions) and self.heels2buttocks_right_down_detector(positions):
            return 'left_up_right_down'
        elif self.heels2buttocks_left_down_detector(positions) and self.heels2buttocks_right_up_detector(positions):
            return 'left_down_right_up'
        else :
            return 'other'

    # -------------------------------------------------------------------------
    #                                                        Méthodes statiques
    # -------------------------------------------------------------------------
    @staticmethod
    def heels2buttocks_left_up_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the left up pose.
        """
        # if landmarks:
        left_heel_hip_dist = calc_distance(landmarks, 27, 23)
        left_knee_hip_dist = calc_distance(landmarks, 25, 23)
        if left_heel_hip_dist < left_knee_hip_dist:
            return True

        return False

    @staticmethod
    def heels2buttocks_right_up_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the right up pose.
        """
        # if landmarks:
        right_heel_hip_dist = calc_distance(landmarks, 28, 24)
        right_knee_hip_dist = calc_distance(landmarks, 26, 24)
        if right_heel_hip_dist < right_knee_hip_dist:
            return True

        return False

    @staticmethod
    def heels2buttocks_left_down_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the left down pose.
        """
        # if landmarks:
        left_heel_hip_dist = calc_distance(landmarks, 27, 23)
        left_knee_hip_dist = calc_distance(landmarks, 25, 23)
        if left_heel_hip_dist > left_knee_hip_dist * 1.5:
            return True

        return False

    @staticmethod
    def heels2buttocks_right_down_detector(landmarks: np.ndarray) -> bool:
        """
        Detects the right down pose.
        """
        # if landmarks:
        right_heel_hip_dist = calc_distance(landmarks, 28, 24)
        right_knee_hip_dist = calc_distance(landmarks, 26, 24)
        if right_heel_hip_dist > right_knee_hip_dist * 1.5:
            return True

        return False

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe Heels2ButtocksDetector *")
    print("-"*80)
    import mediapipe as mp
    # Modèle mediapipe :
    mediapipe_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Capture vidéo :
    cap = cv2.VideoCapture(0)
    # Réglage de la verbosité
    verbose = True
    heels_2_buttocks_detector = Heels2ButtocksDetector(mediapipe_model, cap, verbose)
    heels_2_buttocks_detector.run(objective=4)
