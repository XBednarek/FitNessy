# Import relatif dans le package :
from .posedetector import PoseDetector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calcul_angle,get_points_visibles
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.plankdetector

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
        if self.is_elbow_ankle_aligned_y(positions) and self.check_plank_angle(positions,visibility) :
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
    
    def check_plank_angle(self,position,visibility) :

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

        points_to_check= {
                 "left_wrist": cst.LEFT_WRIST,
                "right_wrist": cst.RIGHT_WRIST,
                "left_shoulder":cst.LEFT_SHOULDER,
                "right_shoulder":cst.RIGHT_SHOULDER,
                "left_hip": cst.LEFT_HIP,
                "right_hip": cst.RIGHT_HIP
           
                
            }
        # choisir le côté détecté
        points_visibles=get_points_visibles(position, visibility, points_indices=points_to_check,image_shape=self.frame.shape)

        if points_visibles and "right_wrist" in points_visibles and "right_shoulder" in points_visibles and "right_hip" in points_visibles:
            pos1, pos2, pos3 = points_visibles["right_wrist"], points_visibles["right_shoulder"], points_visibles["right_hip"] # detecter le coté droit
         
        elif points_visibles and "left_wrist" in points_visibles and "left_shoulder" in points_visibles and "left_hip" in points_visibles:
            pos1, pos2, pos3 = points_visibles["left_wrist"], points_visibles["left_shoulder"], points_visibles["left_hip"] # detecter le coté gauche 
        else:
            pos1 = None
            pos2 = None
            pos3 = None


        if pos1 and pos2 and pos3: # si les trois points sont dectectés, on calcule l'angle
            angle = calcul_angle(pos1, pos2, pos3)
            if 80 < angle < 100:
                return True
            else:
                return False

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
    meditationpose_detector.run(objective=40)
        