# Import relatif dans le package :
from .posedetector import PoseDetector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calcul_angle
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.pushupdetector

# Autres imports
import numpy as np
import cv2
from queue import Queue

class PlankDetector(PoseDetector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False,
                                             show_landmark:bool = False,
                                             windows_name:str = cst.WIN_NAME_HEELS2BUTTOCKS,
                                             frame_queue:Queue|None = None) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose=verbose,
                                               show_landmark=show_landmark,
                                               windows_name=windows_name,
                                               frame_queue=frame_queue,
                                               reward_string = "sec. of plank pose !",
                                               pose_to_keep = "plank",
                                               exo_name = "Plank")
        
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    # Implémentation de la méthode abstraite de la classe mère Detector
    def detect(self, positions: np.ndarray, visibility: np.ndarray) -> str:
        """detect if the person is doing plank exercice"""
        if self.is_elbow_ankle_aligned_y(positions) and self.check_plank_angle(positions) :
            return 'plank'
        else :
            return 'other' 
            
    def is_elbow_ankle_aligned_y(self,position): 
        """Check if the elbows and ankles are approximately aligned along the vertical axis (y),
           to ensure the person is in a proper plank position.

        Args:
            position (list of Landmark): 
            A list of 33 MediaPipe Pose landmarks. Each landmark should have a `.y`.

        Returns:
            bool: 
                True if the elbows and ankles are approximately aligned vertically (within tolerance), 
                False otherwise.
    
        """

        # retriev the y coordinate
        right_elbow = position[cst.RIGHT_ELBOW][1]
        left_elbow  = position[cst.LEFT_ELBOW][1]
        #right_ankle = position[cst.RIGHT_ANKLE][1]
        #left_ankle  = position[cst.LEFT_ANKLE][1]
        right_foot_index = position[cst.RIGHT_FOOT_INDEX][1]
        left_foot_index = position[cst.LEFT_FOOT_INDEX][1]

        tolerance = 0.15
        y_values = [right_foot_index,left_foot_index, right_elbow, left_elbow]

        # chek if ankel and elbow are more or less at the same y
        return max(y_values)-min(y_values) < tolerance
    
    def check_plank_angle(self,position) :

        """Check if the shoulder-hip-knee angles are approximately straight on at least one side
            to verify a proper plank posture.

        Args:
            position (list of Landmark): 
                A list of 33 MediaPipe Pose landmarks. Each landmark should have `.x`, `.y`, 
                `.z` attributes.

        Returns:
            bool: 
                True if the plank is approximately straight on at least one side, False otherwise.

        """

        # retrieve coordinate needed
        right_knee_coordinate = position[cst.RIGHT_KNEE]
        left_knee_coordinate  = position[cst.LEFT_KNEE]
        right_hip_coordinate = position[cst.RIGHT_HIP]
        left_hip_coordinate  = position[cst.LEFT_HIP]
        right_shoulder_coordinate = position[cst.RIGHT_SHOULDER]
        left_shoulder_coordinate  = position[cst.LEFT_SHOULDER]
        right_elbow_coordinate = position[cst.RIGHT_ELBOW]
        left_elbow_coordinate  = position[cst.LEFT_ELBOW]
        

        # calcul angle  shoulder hip knee
        angle_right = calcul_angle(right_elbow_coordinate,right_shoulder_coordinate,right_hip_coordinate)
        angle_left = calcul_angle (left_elbow_coordinate,left_shoulder_coordinate,left_hip_coordinate)

        # tolerance around 180°
        return 80< angle_right < 100 or 80< angle_left < 100

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe PlankDetector *")
    print("-"*80)
    import mediapipe as mp
    # Modèle mediapipe :
    mediapipe_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Capture vidéo :
    cap = cv2.VideoCapture(0)
    # Réglage de la verbosité
    verbose = True
    meditationpose_detector = PlankDetector(mediapipe_model, cap, verbose, show_landmark=True)
    meditationpose_detector.run(objective=4)
        