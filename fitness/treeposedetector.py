# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calc_distance, image2position
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.treeposedetector

# Autres imports
import numpy as np
import cv2

class TreePoseDetector(Detector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False, show_landmark:bool = False, windows_name:str = cst.WIN_NAME_TREE_POSE) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose, show_landmark, windows_name)


    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    
    # Masquage
    def run(self, objective:int) -> float:
        """Run le décompte et renvoie le temps que l'exercice à été
           réalisé"""
        
        move = cst.MOVE_UNKNWON
        tolerance_max = 5
        tolerance = tolerance_max
        min_detect_pose_frames = 5
        detect_pose_frames = 0
        elapsed_time = 0
        
        while  self.cap.isOpened():
        
            # Process de l'image de la webcam
            success = self.read_and_process()
        
            if not success :
                if self.verbose :
                    print("Ignoring empty camera frame.")
                continue

            # Récupération de l'array des positions
            positions = self.getPositions()
            
            # Detect :
            detection = "None"
            if positions is not None :
                detection = self.detect(positions)

                if self.verbose :
                    print(f"{detection =}")

                # Count
                # detection = detected move
                # move = next move to perform
                if detection == "tree_pose":
                    if move == cst.MOVE_UNKNWON :
                        if detect_pose_frames < min_detect_pose_frames:
                            detect_pose_frames += 1
                        elif detect_pose_frames == min_detect_pose_frames:
                            start_timer = cv2.getTickCount()
                            move = cst.MOVE_TREE_POSE
                    elif move == cst.MOVE_TREE_POSE:
                        if tolerance < tolerance_max:
                            tolerance += 1
                elif detection == "other" and move == cst.MOVE_UNKNWON:
                    detect_pose_frames = 0
                elif detection == "other" and move == cst.MOVE_TREE_POSE:
                    tolerance -= 1

            if move == cst.MOVE_TREE_POSE:
                elapsed_ticks = cv2.getTickCount() - start_timer
                elapsed_time = round(elapsed_ticks / cv2.getTickFrequency(), 2)
            
            # Show counter :
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 0)
            thickness = 1
            
            # Define positions for the text
            position = (10, 20)
            position2 = (10, 40) 
            position3 = (10, 60)

            text_count = f"Tree pose #: {elapsed_time} sec."
            text_status = f"Move status: {move}"
            text_lastdetect = f"Last: {detection}"
            cv2.putText(self.image, text_count, position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(self.image, text_status, position2, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(self.image, text_lastdetect, position3, font, font_scale, color, thickness, cv2.LINE_AA)

            if elapsed_time >= objective :
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 95, 31)
                thickness = 2
                text1 = "Congratulations !"
                position1 = (200, 100)
                text2 = "you performed"
                position2 = (220, 150)
                text3 = f"{objective} sec."
                position3 = (250, 225)
                text4 = "of the Tree pose !"
                position4 = (180, 275)
                cv2.putText(self.image, text1, position1, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(self.image, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(self.image, text3, position3, font, font_scale+1, color, thickness+1, cv2.LINE_AA)
                cv2.putText(self.image, text4, position4, font, font_scale, color, thickness, cv2.LINE_AA)

            if tolerance == 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 95, 31)
                thickness = 2
                text1 = "Booooooooooooh !"
                position1 = (200, 100)
                text2 = "you failed at"
                position2 = (220, 150)
                text3 = f"{elapsed_time} sec."
                position3 = (220, 225)
                text4 = "of the Tree pose !"
                position4 = (180, 275)
                cv2.putText(self.image, text1, position1, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(self.image, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(self.image, text3, position3, font, font_scale+1, color, thickness+1, cv2.LINE_AA)
                cv2.putText(self.image, text4, position4, font, font_scale, color, thickness, cv2.LINE_AA)

            # Affichage
            self.imshow()

            # Quitte si l'objectif est atteint
            if elapsed_time >= objective :
                cv2.waitKey(5000)
                break

            if tolerance == 0:
                cv2.waitKey(5000)
                break

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # On quitte
        self.close()

        return elapsed_time


    # Masquage
    def detect(self, positions: np.ndarray) -> str:
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
    treepose_detector = TreePoseDetector(mediapipe_model, cap, verbose)
    treepose_detector.run(objective=4)
