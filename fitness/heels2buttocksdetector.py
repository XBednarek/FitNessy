# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import calc_distance, image2position
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.heels2buttocksdetector

# Autres imports
import numpy as np
import cv2

class Heels2ButtocksDetector(Detector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose)


    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    
    # Masquage
    def run(self, objective:int) -> float:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        
        heels_butt_counter = 0
        move = cst.MOVE_UNKNWON
        
        while  self.cap.isOpened():
        
            success, image = self.cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue     # If loading a video, use 'break' instead of 'continue'.

            # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process
            positions = image2position(image, mediapipe_model=self.mediapipe_model, show=False)
            
            # Detect :
            detection = "None"
            if positions is not None :
                detection = self.detect(positions)

                if self.verbose :
                    print(f"{detection =}")

                # Count
                # detection = detected move
                # move = next move to perform
                if detection == "left_up_right_down":
                    if move == cst.MOVE_UNKNWON :
                        move = cst.MOVE_LEFT_DOWN_RIGHT_UP
                    elif move == cst.MOVE_LEFT_UP_RIGHT_DOWN :
                        heels_butt_counter += 0.5
                        move = cst.MOVE_LEFT_DOWN_RIGHT_UP
                elif detection == "left_down_right_up":
                    if move == cst.MOVE_UNKNWON :
                        move = cst.MOVE_LEFT_UP_RIGHT_DOWN
                    elif move == cst.MOVE_LEFT_DOWN_RIGHT_UP :
                        heels_butt_counter += 0.5
                        move = cst.MOVE_LEFT_UP_RIGHT_DOWN

            
            # Show counter :
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 0)
            thickness = 1
            
            # Define positions for the text
            position = (10, 20)
            position2 = (10, 40) 
            position3 = (10, 60)

            text_count = f"Heels-butt #: {heels_butt_counter}"
            text_status = f"Move status: {move}"
            text_lastdetect = f"Last: {detection}"
            cv2.putText(image, text_count, position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, text_status, position2, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, text_lastdetect, position3, font, font_scale, color, thickness, cv2.LINE_AA)

            if heels_butt_counter >= objective :
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 95, 31)
                thickness = 2
                text1 = "Congratulations !"
                position1 = (200, 100)
                text2 = "you performed"
                position2 = (220, 150)
                text3 = f"{objective}"
                position3 = (300, 225)
                text4 = "heels-to-buttocks !"
                position4 = (180, 275)
                cv2.putText(image, text1, position1, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(image, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(image, text3, position3, font, 2, color, 3, cv2.LINE_AA)
                cv2.putText(image, text4, position4, font, font_scale, color, thickness, cv2.LINE_AA)

            # Convertion avant affichage
            result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow(cst.WIN_NAME_HEELS2BUTTOCKS, result_image)

            if heels_butt_counter >= objective :
                cv2.waitKey(5000)
                break

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Destruction de la fenetre
        cv2.destroyWindow(cst.WIN_NAME_HEELS2BUTTOCKS)

        return heels_butt_counter


    # Masquage
    def detect(self, positions: np.ndarray) -> str:
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
