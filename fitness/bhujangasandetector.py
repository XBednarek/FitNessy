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
import time



class Bhujangasandetector(Detector):


    def __init__(self, mediapipe_model, cap, verbose:bool = False) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose)
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    # extract les points important pour detecter le mouvement de youga

    def extract_points(self,result, image_shape:tuple) -> np.ndarray:
        # durée totale en seconds

        h,w, _ = image_shape
        points_yoga={}

        if not result.pose_landmarks:
            return None
        
        # lancer notre module de pose 

        landmarks=result.pose_landmarks.landmark # 33 points detecter au niveau en squelette
        mp_pose=mp.solutions.pose

        # trouver les chevilles et les hanches et la tete

        # 1) detections les hanches gauche et droit 
        left_hip  = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        #la visibilité des hanches dans quel coté visible gauche et droit 
        if left_hip.visibility > 0.5: # on regarde si la hanche est suffisament visible 
            points_yoga["left_hip"] = (int(left_hip.x * w), int(left_hip.y * h)) # normaliser la hauteur et largeur de points par le W et h d'image
        if right_hip.visibility > 0.5:
            points_yoga["right_hip"] = (int(right_hip.x * w), int(right_hip.y *h))


        # 2) detection des chevilles
        left_ankle  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        if left_ankle.visibility > 0.5:  
            points_yoga["left_ankle"] = (int(left_ankle.x * w), int(left_ankle.y * h)) 
        if right_ankle.visibility > 0.5:
            points_yoga["right_ankle"] = (int(right_ankle.x * w), int(right_ankle.y *h))

        # 3) detection des epaules
        left_shoulder  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if left_shoulder.visibility > 0.5:  
            points_yoga["left_shoulder"] = (int(left_shoulder.x * w), int(left_shoulder.y * h)) 
        if right_ankle.visibility > 0.5:
            points_yoga["right_shoulder"] = (int(right_shoulder.x * w), int(right_shoulder.y *h))
        
        # 4) detection des yeux ( reference de la position de la tete )

        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        points_yoga["top_head"] = (int(nose.x * w), int(nose.y *h))


        return points_yoga


 
    def run(self, objective:int) -> int:
        """Run compter le temps que une persone a fai ce mouvement et renvoie le temps finale de l'exercice de yoga à été
        réalisé"""
        self.timer_start= None # le moment ou la posture 'up' commence
        self.timer_duration=0
        mp_pose=mp.solutions.pose # initialiser la classe de pose qui detecte les points de la squellete
        pose = mp_pose.Pose() # initialiser l'objet Pose pour pouvoir traiter des images
        # result= pose.process(image) # traiter l'image et detecter les points clés du corps

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

            points = self.extract_points(results, frame.shape) # extraire les points important dans notre image
            
            # choisir le côté détecté
            
            if points and "right_hip" in points and "right_shoulder" in points and "right_ankle" in points:
                hip, shoulder, ankle = points["right_hip"], points["right_shoulder"], points["right_ankle"] # detecter le coté droit
            elif points and "left_hip" in points and "left_shoulder" in points and "left_ankle" in points :
                hip, shoulder, ankle = points["left_hip"], points["left_shoulder"], points["left_ankle"] # detecter le coté gauche 
            else:
                hip=None
                shoulder=None
                ankle=None
             
            
            if hip and shoulder and ankle : # si les trois points sont dectectés, on calcule l'angle
                angle_soulder_hip_ankle = calcul_angle(shoulder,hip,ankle) # calculer l'angle entre les epaules, les hanches et les chevilles

                if  angle_soulder_hip_ankle <= 100:
                    self.stage = "up"
                    if self.timer_start is None:
                        self.timer_start=time.time()

                else:
                    if self.timer_start is not None:
                        self.timer_duration = time.time() - self.timer_start
    
                cv2.putText(frame, str(int(angle_soulder_hip_ankle)), (int(hip[0])+20, int(hip[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                # afficher compteur
                cv2.putText(frame, f'aconda yoga Timer: {round(self.timer_duration,2)}', (50,100),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (18,18,18), 3)


                # dessiner les points
                for p in [hip,shoulder, ankle]:
                    if p is not None:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 10, (0,255,255), -1)

            cv2.imshow(cst.WIN_NAME_COBRA, frame)


            if cv2.waitKey(5) & 0xFF == ord('q'):  # q pour quitter
                break
    
            if self.timer_duration >= objective:
                break

        # Destruction de la fenetre
        cv2.destroyWindow(cst.WIN_NAME_COBRA)


        return self.timer_duration

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe SquatDetector *")
    print("-"*80)

    # import cv2
    # import mediapipe as mp

    # # Initialiser la webcam
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Erreur : impossible d'ouvrir la webcam")
    #     exit()

    # # Créer une instance de Bhujangasandetector
    # detector = Bhujangasandetector(mediapipe_model=mp.solutions.pose.Pose(), cap=cap, verbose=True)

    # # Objectif : temps en secondes pour maintenir la posture
    # objectif_temps = 10

    # # Lancer le détecteur (compteur et timer)
    # total_time = detector.run(objective=objectif_temps)

    # print(f"Exercice terminé ! Temps total enregistré : {total_time} secondes")

    # # Libération de la webcam
    # cap.release()
    # cv2.destroyAllWindows()

