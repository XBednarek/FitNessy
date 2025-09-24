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

        landmarks=result.pose_landmarks.landmark  # liste de 33 points de squelette
        mp_pose=mp.solutions.pose # creer un reference de module de pose
        

        
        # detections les hanches gauche et droit 
        left_hip  = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        #la visibilité des hanches dans quel coté visible gauche et droit 
        if left_hip.visibility > 0.5: # on regarde si la hanche est suffisament visible 
            key_points["left_hip"] = (int(left_hip.x * w), int(left_hip.y * h)) # normaliser la hauteur et largeur de points par le W et h d'image
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
        

        # si l'image qui prend en milieu
        # garder que le mouvement des hanches le plus fiable car les genous et les chevilles sont moins fiable 
        # pour les hanches : point central si les deux côtés sont visibles
        if "left_hip" in key_points and "right_hip" in key_points:
            lx, ly = key_points["left_hip"]
            rx, ry = key_points["right_hip"]
            key_points["hip_center"] = ((lx + rx) / 2, (ly + ry) / 2) # position horizontale moyenne des deux hanches (hauteur,largeur)
        elif "left_hip" in key_points:
            key_points["hip_center"] = key_points["left_hip"]
        elif "right_hip" in key_points:
            key_points["hip_center"] = key_points["right_hip"]

            
        return key_points
    
    def run(self, objective:int) -> int:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        
        mp_pose=mp.solutions.pose # initialiser la classe de pose qui detecte les points de la squellete
        pose = mp_pose.Pose() # initialiser l'objet Pose pour pouvoir traiter des images
        # result= pose.process(image) # traiter l'image et detecter les points clés du corps

        self.counter = 0

        if not self.cap.isOpened():
            print("Erreur : impossible d'ouvrir la webcam")
            return

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # convertir en RGB pour Mediapipe

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            points = self.extract_position(results, frame.shape) # extraire les points important dans notre image
            
            # choisir le côté détecté
            
            if points and "right_hip" in points and "right_knee" in points and "right_ankle" in points:
                hip, knee, ankle = points["right_hip"], points["right_knee"], points["right_ankle"] # detecter le coté droit
            elif points and "left_hip" in points and "left_knee" in points and "left_ankle" in points:
                hip, knee, ankle = points["left_hip"], points["left_knee"], points["left_ankle"] # detecter le coté gauche 
            else:
                hip=None
                knee=None
                ankle=None
            
            if hip and knee and ankle: # si les trois points sont dectectés, on calcule l'angle
                angle = calcul_angle(hip, knee, ankle)

                if angle < 130:
                    self.stage = "down"
                elif angle > 160 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
    
                cv2.putText(frame, str(int(angle)), (int(knee[0])+20, int(knee[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                # afficher compteur
                cv2.putText(frame, f'squat_counter: {self.counter}', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                # dessiner les points
                for p in [hip, knee, ankle]:
                    if p is not None:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 10, (0,255,255), -1)

            cv2.imshow(cst.WIN_NAME_SQUATS, frame)


            if cv2.waitKey(5) & 0xFF == ord('q'):  # q pour quitter
                break
      
            if self.counter >= objective:
                break

        # Destruction de la fenetre
        cv2.destroyWindow(cst.WIN_NAME_SQUATS)


        return self.counter

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe SquatDetector *")
    print("-"*80)
    