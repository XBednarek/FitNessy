# Import relatif dans le package :
from .movedetector import MoveDetector
from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.squatdetector
from .tools import calcul_angle, get_points_visibles

# Autres imports
import numpy as np
import mediapipe as mp
import cv2

class SquatDetector(MoveDetector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False, 
                                            show_landmark:bool = False, 
                                            windows_name:str = cst.WIN_NAME_SQUATS) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose=verbose, 
                                                show_landmark=show_landmark,
                                                windows_name=windows_name,
                                                reward_string = "squat !",
                                               movement_cycle = ["up", "down"],
                                               exo_name = "squat")
        

    # -------------------------------------------------------------------------
    def detect(self, positions: np.ndarray, visibility: np.ndarray) -> str:
        """Détecte une position"""

        points_to_check= {
                "left_hip": cst.LEFT_HIP,
                "right_hip": cst.RIGHT_HIP,
                "left_knee":cst.LEFT_KNEE,
                "right_knee":cst.RIGHT_KNEE,
                "left_ankle": cst.LEFT_ANKLE,
                "right_ankle": cst.RIGHT_ANKLE
                
            }
        # choisir le côté détecté
        points_visibles=get_points_visibles(positions, visibility, points_indices=points_to_check,image_shape=self.frame.shape)
        
        if points_visibles and "right_hip" in points_visibles and "right_knee" in points_visibles and "right_ankle" in points_visibles:
            pos1, pos2, pos3 = points_visibles["right_hip"], points_visibles["right_knee"], points_visibles["right_ankle"] # detecter le coté droit
         
        elif points_visibles and "left_hip" in points_visibles and "left_knee" in points_visibles and "left_ankle" in points_visibles:
            pos1, pos2, pos3 = points_visibles["left_hip"], points_visibles["left_knee"], points_visibles["left_ankle"] # detecter le coté gauche 
        else:
            pos1 = None
            pos2=None
            pos3=None

        if pos1 and pos2 and pos3: # si les trois points sont dectectés, on calcule l'angle
            angle = calcul_angle(pos1, pos2, pos3)
            if angle < 130:
                return "down"
            elif angle > 160 :
                return "up"
            else:
                return "other"
        
        
if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe SquatDetector *")
    print("-"*80)

    import cv2
    import mediapipe as mp

    
    # Modèle mediapipe :
    mediapipe_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Capture vidéo :
    cap = cv2.VideoCapture(0)
    # Réglage de la verbosité
    verbose = True
    squat_detector = SquatDetector(mediapipe_model, cap, verbose, show_landmark=True)
    squat_detector.run(objective=4)
