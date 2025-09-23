# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import image2position,calc_distance
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.pushupdetector

# Autres imports
import numpy as np
import cv2
import joblib

class PushUpDetector(Detector):

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
    def run(self, objective:int, methode = "Analytics") -> int:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        push_up_counter = 0
        move = cst.MOVE_UNKNWON
        while  self.cap.isOpened():
        
            success, image = self.cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue     # If loading a video, use 'break' instead of 'continue'.

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process
            positions = image2position(image, mediapipe_model=self.mediapipe_model, show=False)

            # Detect :
            detection = "None"
            if positions is not None :
                detection = self.detect(positions,methode)

                if self.verbose :
                    print(f"{detection =}")

                # Count
                if detection == "up" :
                    if move == cst.MOVE_UNKNWON :
                        move = cst.MOVE_DOWN
                    elif move == cst.MOVE_UP :
                        push_up_counter += 0.5
                        move = cst.MOVE_DOWN
                elif detection == "down" :
                    if move == cst.MOVE_UNKNWON :
                        move = cst.MOVE_UP
                    elif move == cst.MOVE_DOWN :
                        push_up_counter += 0.5
                        move = cst.MOVE_UP

            # Show counter :
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 0)
            thickness = 1
            
            # Define positions for the text
            position = (10, 20)
            position2 = (10, 40) 
            position3 = (10, 60)

            # Show counter :
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 0)
            thickness = 1
            
            # Define positions for the text
            position = (10, 20)
            position2 = (10, 40) 
            position3 = (10, 60)

            text_count = f"Pushup #: {push_up_counter}"
            text_status = f"Move status: {move}"
            text_lastdetect = f"Last: {detection}"
            cv2.putText(image, text_count, position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, text_status, position2, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, text_lastdetect, position3, font, font_scale, color, thickness, cv2.LINE_AA)

            # Convertion avant affichage
            result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow(cst.WIN_NAME_PUSHUP, result_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            if push_up_counter >= objective :
                break

        # Destruction de la fenetre
        cv2.destroyWindow(cst.WIN_NAME_PUSHUP)

        return push_up_counter


       # Masquage
    def detect(self, positions: np.ndarray, methode: str = "Analytics") -> str :
        """ Get the array of positions and detect if the position is "up",
            "down" or "other". 
            We can choose if we do the detection with or without ML

        Args:
            positions (np.ndarray): Array as given by image2position
            model (str, optional): "ML" or "Not_ML", to choose the way the classification is done

        Returns:
            bool (str): "up", "down" or "other" 
        """

        if methode == "KNN" :
            return self.detect_ML(positions)
        elif methode == "Analytics":
            return self.detect_analytics(positions)
        

    def detect_ML(self, positions: np.ndarray) -> str:
        """Détecte une position"""
        if positions is None:
            return "unknown"
        
        model_knn = joblib.load("../model/knn_pompe_model.pkl")
        usefull_point = [
                        cst.LEFT_SHOULDER.value, cst.LEFT_ELBOW.value, cst.LEFT_WRIST.value,
                        cst.RIGHT_SHOULDER.value, cst.RIGHT_ELBOW.value, cst.RIGHT_WRIST.value
                        ]
        X = np.array(positions)[usefull_point]
        X_new = X.flatten().reshape(1, -1)
        #prediction = model_knn.predict(X_new)
        # retrieve the probability
        proba = model_knn.predict_proba(X_new)[0]
        # array with up other and down
        classes = model_knn.classes_
        # indix of the best proba
        best_idx = np.argmax(proba)
        # what is the class of the best proba
        best_class = classes[best_idx]
        # value of best proba
        best_proba = proba[best_idx]

        # Threshold
        thresholds = {'up': 0.8, 'down': 0.6}

        if best_class in thresholds and best_proba >= thresholds[best_class]:
            return best_class
        else:
            return "other"
        
    def detect_analytics(self, positions: np.ndarray) -> str:
        """Détecte une position"""
        if self.push_up_up_detector(positions):
            return 'up'
        elif self.push_up_down_detector(positions):
            return 'down'
        else :
            return 'other'

    def push_up_up_detector(self,landmarks: np.ndarray) -> bool:
        """
        Detects the up pose.
        """
        # if landmarks:
        left_wrist_shoulder_dist = calc_distance(landmarks, 15, 11)
        left_wrist_elbow_dist = calc_distance(landmarks, 15, 13)
        right_wrist_shoulder_dist = calc_distance(landmarks, 16, 12)
        right_wrist_elbow_dist = calc_distance(landmarks, 16, 14)
        if left_wrist_shoulder_dist > left_wrist_elbow_dist * 1.5 and right_wrist_shoulder_dist > right_wrist_elbow_dist * 1.5:
            return True

        return False

    def push_up_down_detector(self,landmarks: np.ndarray) -> bool:
        """
        Detects the up pose.
        """
        # if landmarks:
        left_wrist_shoulder_dist = calc_distance(landmarks, 15, 11)
        left_wrist_elbow_dist = calc_distance(landmarks, 15, 13)
        right_wrist_shoulder_dist = calc_distance(landmarks, 16, 12)
        right_wrist_elbow_dist = calc_distance(landmarks, 16, 14)
        if left_wrist_shoulder_dist < left_wrist_elbow_dist and right_wrist_shoulder_dist < right_wrist_elbow_dist:
            return True

        return False


if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe PushUpDetector *")
    print("-"*80)
