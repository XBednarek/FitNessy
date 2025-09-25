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

class SquatDetector(Detector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False, show_landmark:bool = False, windows_name:str = cst.WIN_NAME_SQUATS) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose, show_landmark, windows_name)
        self.stage= None
        self.counter=0

    # -------------------------------------------------------------------------
    
    
    def run(self, objective:int) -> float:
        """
        Run le décompte et renvoie le nombre de fois que l'exercice à été réalisé

        Args:
            objective (int): le nombre de mouvement

        Returns:
            float: le totale des mouvements
        """
        
        self.counter = 0

        while self.cap.isOpened():

            # Process de l'image de la webcam
            success = self.read_and_process()
        
            if not success :
                if self.verbose :
                    print("Ignoring empty camera frame.")
                continue



            

                # afficher compteur
                cv2.putText(self.image, f'squat_counter: {self.counter}', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                # dessiner les points
                for p in [hip, knee, ankle]:
                    if p is not None:
                        cv2.circle(self.image, (int(p[0]), int(p[1])), 10, (0,255,255), -1)

            # Affichage
            self.imshow()

            if cv2.waitKey(5) & 0xFF == ord('q'):  # q pour quitter
                break
      
            if self.counter >= objective:
                break

        # On quitte
        self.close()

        return self.counter
    
    def detect(self, positions: np.ndarray) -> str:
        """Détecte une position"""
        if self.squate_detector_up(positions):
            return 'up'
        elif self.squate_detector_down(positions):
            return 'down'
        else :
            return 'other'

    def squate_detector_up(self,):

    # definir les positions importantes :
        points_to_check= {
                "left_hip": cst.LEFT_HIP,
                "right_hip": cst.RIGHT_HIP,
                "left_knee":cst.LEFT_KNEE,
                "right_knee":cst.RIGHT_KNEE,
                "left_ankle": cst.LEFT_ANKLE,
                "right_ankle": cst.RIGHT_ANKLE
                
            }
        
        # choisir le côté détecté
        points_visibles=self.get_points_visibles(points_indices=points_to_check,image_shape=self.frame.shape)

        if points_visibles and "right_hip" in points_visibles and "right_knee" in points_visibles and "right_ankle" in points_visibles:
            hip, knee, ankle = points_visibles["right_hip"], points_visibles["right_knee"], points_visibles["right_ankle"] # detecter le coté droit
        elif points_visibles and "left_hip" in points_visibles and "left_knee" in points_visibles and "left_ankle" in points_visibles:
            hip, knee, ankle = points_visibles["left_hip"], points_visibles["left_knee"], points_visibles["left_ankle"] # detecter le coté gauche 
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

            cv2.putText(self.image, str(int(angle)), (int(knee[0])+20, int(knee[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


        return
    




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
    detector = SquatDetector(mediapipe_model=mp.solutions.pose.Pose(), cap=cap, verbose=True)

    # Objectif : temps en secondes pour maintenir la posture
    objectif_temps = 3

    # Lancer le détecteur (compteur et timer)
    total_time = detector.run(objective=objectif_temps)

    print(f"Exercice terminé ! Temps total enregistré : {total_time} secondes")

    # Libération de la webcam
    cap.release()
    cv2.destroyAllWindows()
