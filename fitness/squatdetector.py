# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.squatdetector
from .tools import calcul_angle

# Autres imports
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

class SquatDetector(Detector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose)
        self.stage= None
        self.counter=0
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    # extract les points important pour detecter le mouvement de squat
    def extract_position(self,result,frame_shape :tuple) -> np.ndarray:
        """
        extraire les positions (hanche, genou et cheville).
        """
        h,w,_=frame_shape
        key_points={}
        if not result.pose_landmarks:
            # Draw landmarks on the image
            #mp.solutions.drawing_utils.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            return None
        
        #retourner les hanches, les genou et les chevilles

        landmarks=result.pose_landmarks.landmark
        mp_pose=mp.solutions.pose
        

        
        # detections les hanches 
        left_hip  = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        #la visibilité des hanches dans quel coté visible
        if left_hip.visibility > 0.5:
            key_points["left_hip"] = (int(left_hip.x * w), int(left_hip.y * h))
        if right_hip.visibility > 0.5:
            key_points["right_hip"] = (int(right_hip.x * w), int(right_hip.y *h))


        # detections les genoux
        left_knee    = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee    = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]


        # la visibilité des genoux coté gauche ou droit
        if left_knee.visibility > 0.5:
            key_points["left_knee"] = (int(left_knee.x * w), int(left_knee.y * h))
        if right_knee.visibility > 0.5:
            key_points["right_knee"] = (int(right_knee.x * w), int(right_knee.y *h))
        
        # detections des chevilles

        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        if left_ankle.visibility > 0.5:
            key_points["left_ankle"] = (left_ankle.x * w, left_ankle.y * h)
        if right_ankle.visibility > 0.5:
            key_points["right_ankle"] = (right_ankle.x * w, right_ankle.y * h)
        

        # si limage qui prend en milieu

        # pour les hanches : point central si les deux côtés sont visibles
        if "left_hip" in key_points and "right_hip" in key_points:
            lx, ly = key_points["left_hip"]
            rx, ry = key_points["right_hip"]
            key_points["hip_center"] = ((lx + rx) / 2, (ly + ry) / 2)
        elif "left_hip" in key_points:
            key_points["hip_center"] = key_points["left_hip"]
        elif "right_hip" in key_points:
            key_points["hip_center"] = key_points["right_hip"]

        # pour les genoux : point central si les deux côtés sont visibles
        if "left_knee" in key_points and "right_knee" in key_points:
            lx, ly = key_points["left_knee"]
            rx, ry = key_points["right_knee"]
            key_points["knee_center"] = ((lx + rx) / 2, (ly + ry) / 2)
        elif "left_knee" in key_points:
            key_points["knee_center"] = key_points["left_knee"]
        elif "right_knee" in key_points:
            key_points["knee_center"] = key_points["right_knee"]

        # pour les chevilles : même logique
        if "left_ankle" in key_points and "right_ankle" in key_points:
            lx, ly = key_points["left_ankle"]
            rx, ry = key_points["right_ankle"]
            key_points["ankle_center"] = ((lx + rx) / 2, (ly + ry) / 2)
        elif "left_ankle" in key_points:
            key_points["ankle_center"] = key_points["left_ankle"]
        elif "right_ankle" in key_points:
            key_points["ankle_center"] = key_points["right_ankle"]
            
        return key_points
    

    # Masquage
    def detect(self, positions: np.ndarray) -> str:
        """Détecte une position dans les deux cotés droit et gauche"""

        if positions is None or len(positions)<3:
            return "non reconnu"
        
        hip = knee = ankle = None
    
        if positions and "right_hip" in positions and "right_knee" in positions and "right_ankle" in positions:
            hip, knee, ankle = positions["right_hip"], positions["right_knee"], positions["right_ankle"]
        elif positions and "left_hip" in positions and "left_knee" in positions and "left_ankle" in positions:
            hip, knee, ankle = positions["left_hip"], positions["left_knee"], positions["left_ankle"]
        else:
            hip = knee = ankle = None

        if hip and knee and ankle:
            angle = calcul_angle(hip, knee, ankle)

            if angle < 100:
                self.stage = "down"
                return "down"
            elif angle > 160 and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                return "up"
        return " non reconnu"

    def run(self, objective:int) -> int:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        
        mp_pose=mp.solutions.pose
        pose = mp_pose.Pose()
        self.counter = 0
        

        if not self.cap.isOpened():
            print("Erreur : impossible d'ouvrir la webcam")
            return

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip horizontal et convertir en RGB pour Mediapipe
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            points = self.extract_position(results, frame.shape)
            self.detect(points)

            
            hip=knee=ankle=None
            # choisir le côté détecté
            if points and "right_hip" in points and "right_knee" in points and "right_ankle" in points:
                hip, knee, ankle = points["right_hip"], points["right_knee"], points["right_ankle"]
            elif points and "left_hip" in points and "left_knee" in points and "left_ankle" in points:
                hip, knee, ankle = points["left_hip"], points["left_knee"], points["left_ankle"]

            if hip and knee and ankle:
                angle = calcul_angle(hip, knee, ankle)

                cv2.putText(frame, str(int(angle)), (int(knee[0])+20, int(knee[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                # afficher compteur
                cv2.putText(frame, f'squat_counter: {self.counter}', (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                # dessiner les points
                for p in [hip, knee, ankle]:
                    if p is not None:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 10, (0,255,255), -1)

            cv2.imshow("squat counter", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):  # q pour quitter
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return self.counter


if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe SquatDetector *")
    print("-"*80)
    from .squatdetector import SquatDetector
    import mediapipe as mp
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np



    mp_pose=mp.solutions.pose
    pose_model= mp_pose.Pose()

    detector = SquatDetector(mediapipe_model=pose_model, cap=0, verbose=True)

    # Lancer le compteur

    total_squats = detector.run(objective=10)  # objective est optionnel
    print(f"Nombre total de squats réalisés : {total_squats}")