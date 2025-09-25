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

class Cobradetector(Detector):


    def __init__(self, mediapipe_model, cap, verbose:bool = False, show_landmark:bool = False, windows_name:str = cst.WIN_NAME_COBRA) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose, show_landmark, windows_name)

    # -------------------------------------------------------------------------
 
    def run(self, objective:int) -> float:
        """
        Run : compter le temps d'un mouvement et renvoie le temps finale de l'exercice de yoga à été
        réalisé 

        Args:
            objective (int): le temps que il faut attient

        Returns:
            float: renvoie le temps finale
        """
        self.timer_start= None # le moment ou la posture 'up' commence
        self.timer_duration=0

        while self.cap.isOpened():

            # Process de l'image de la webcam:
            success = self.read_and_process()
        
            if not success :
                if self.verbose :
                    print("Ignoring empty camera frame.")
                continue
            

            # detection via la visibility :
            points_to_check= {
                    "left_hip": cst.LEFT_HIP,
                    "right_hip": cst.RIGHT_HIP,
                    "left_ankle": cst.LEFT_ANKLE,
                    "right_ankle": cst.RIGHT_ANKLE,
                    "left_shoulder":cst.LEFT_SHOULDER,
                    "right_shoulder":cst.RIGHT_SHOULDER
                }
            
            points_visibles=self.get_points_visibles(points_indices=points_to_check,image_shape=self.frame.shape)
            
            # quel coté est detecter ?
            if points_visibles and "right_hip" in points_visibles and "right_shoulder" in points_visibles and "right_ankle" in points_visibles:
                hip, shoulder, ankle = points_visibles["right_hip"], points_visibles["right_shoulder"], points_visibles["right_ankle"] #  le coté droit
            elif points_visibles and "left_hip" in points_visibles and "left_shoulder" in points_visibles and "left_ankle" in points_visibles :
                hip, shoulder, ankle = points_visibles["left_hip"], points_visibles["left_shoulder"], points_visibles["left_ankle"] #  le coté gauche 
            else:
                hip=None
                shoulder=None
                ankle=None # au cas quand il detecte rien == None
             
            
            if hip and shoulder and ankle : # si les trois points sont dectectés, on calcule l'angle
                angle_soulder_hip_ankle = calcul_angle(shoulder,hip,ankle) # calculer l'angle entre les epaules, les hanches et les chevilles

                if  angle_soulder_hip_ankle <= 140:
                    self.stage = "up"
                    if self.timer_start is None:
                        self.timer_start=time.time()

                else:
                    if self.timer_start is not None:
                        self.timer_duration = time.time() - self.timer_start
    
                cv2.putText(self.image, str(int(angle_soulder_hip_ankle)), (int(hip[0])+20, int(hip[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                # afficher compteur
                cv2.putText(self.image, f'aconda yoga Timer: {round(self.timer_duration,2)}', (50,100),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (18,18,18), 3)


                # dessiner les points
                for p in [hip, shoulder, ankle]:
                    if p is not None:
                        cv2.circle(self.image, (int(p[0]), int(p[1])), 10, (0,255,255), -1)

            # Affichage
            self.imshow()

            if cv2.waitKey(5) & 0xFF == ord('q'):  # q pour quitter
                break
    
            if self.timer_duration >= objective:
                break

        # On quitte
        self.close()

        return self.timer_duration

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe SquatDetector *")
    print("-"*80)

    import cv2
    import mediapipe as mp

    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam")
        exit()

    # Créer une instance de Bhujangasandetector
    detector = Cobradetector(mediapipe_model=mp.solutions.pose.Pose(), cap=cap, verbose=True)

    # Objectif : temps en secondes pour maintenir la posture
    objectif_temps = 10

    # Lancer le détecteur (compteur et timer)
    total_time = detector.run(objective=objectif_temps)

    print(f"Exercice terminé ! Temps total enregistré : {total_time} secondes")

    # Libération de la webcam
    cap.release()
    cv2.destroyAllWindows()

