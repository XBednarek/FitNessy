# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.posedetector

# Autres imports
import numpy as np
import cv2
import time
from queue import Queue

class PoseDetector(Detector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False,
                                             show_landmark:bool = False,
                                             windows_name:str = cst.WIN_NAME_DEFAULT,
                                             reward_string:str = "",
                                             fast_api_queues:tuple[Queue, Queue]|None = None,
                                             pose_to_keep:str="",
                                             exo_name: str="") -> None:
        """Constructeur d'un détecteur de mouvement"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose=verbose,
                                               show_landmark=show_landmark,
                                               windows_name=windows_name,
                                               reward_string=reward_string,
                                               fast_api_queues=fast_api_queues)

        # Position à maintenir
        self.pose_to_keep = pose_to_keep

        # Compteur (en secondes):
        self.time_counter_s = 0

        # Objectif (en secondes)
        self.objective_s = None

        # Tolerance max (en secondes):
        self.tol_max_s = 5

        # Tolerance (en secondes):
        self.tol_s = self.tol_max_s

        # Nom de l'exercice :
        self.exo_name = exo_name

    
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    # Implémentation de la méthode abstraite de la classe mère Detector
    # Masquage
    def run(self, objective:int) -> float:
        """Run le décompte et renvoie le temps que l'exercice à été
           réalisé"""
        
        self.objective_s = objective
        
        start_detect_time = None
        start_tol_time = None
        last_time_counter_s = self.time_counter_s
        last_tol_s = self.tol_s

        flag = True   
            
        while self.cap.isOpened():

            # Process de l'image de la webcam
            success = self.read_and_process()
        
            if not success :
                if self.verbose :
                    print("Ignoring empty camera frame.")
                continue

            # Récupération de l'array des positions
            positions = self.getPositions()

            # Récupération de l'array des visibilités
            visibility = self.getVisibility()
            
            if positions is not None :

                # Détection :
                detected_pose = self.detect(positions, visibility)

                # Affichage
                if self.verbose :
                    print(f"Pose détectée : {detected_pose}")

                # Si on detecte la position attendue
                if detected_pose == self.pose_to_keep :

                    # On retient la valeur du précédent décompte
                    if flag :
                        last_tol_s = self.tol_s
                        flag = False

                    # On n'est pas en mode tolérance
                    start_tol_time = None

                    # Si on n'a pas encore démarrer le compteur, on le démarre
                    if start_detect_time is None :
                        start_detect_time = time.time()

                    # On mets à jour le compteur !
                    self.time_counter_s = time.time() - start_detect_time + last_time_counter_s

                    # Et on réaugmente la tolérance tant qu'on tient la pause
                    self.tol_s = time.time() - start_detect_time + last_tol_s

                    # Mais on cap à la tolérance max
                    self.tol_s = min(self.tol_s, self.tol_max_s)

                # Si on ne détecte pas la position attendue
                else  :

                    if not flag :
                        last_tol_s = self.tol_s
                        flag = True

                    # On retient la valeur du précédent décompte
                    last_time_counter_s = self.time_counter_s

                    # On n'est pas en mode detection
                    start_detect_time = None

                    if self.time_counter_s > 0 :
                        # Si on n'a pas encore démarrer le compteur, on le démarre
                        if start_tol_time is None :
                            start_tol_time = time.time()

                        # On décrémente la tolérance
                        self.tol_s = last_tol_s - (time.time() - start_tol_time)
            
            # Monitoring du score ou écran de féliciation
            if self.time_counter_s >= objective :
                self.congrate(objective)
            elif self.tol_s < 0 :
                self.booo()
            else :
                self.displayScore()

            # Affichage
            self.imshow()

            # Quitte si on a épuisé la tolérance
            if self.tol_s < 0 :
                cv2.waitKey( cst.CONGRATS_DURATION_S * 1000)
                break
            
            # Quitte si on appuie sur q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # Quitte si l'objectif est atteint
            if self.time_counter_s >= objective :
                cv2.waitKey( cst.CONGRATS_DURATION_S * 1000)
                break

            # Vérifier s'il y a des actions à faire via FastAPI
            if self.action_queue is not None and not self.action_queue.empty():
                action = self.action_queue.get()
                if self.verbose :
                    print(f"Action reçue: {action}")  # Ici vous faites ce que vous voulez
                if action=="quit":
                    break
                else :
                    # Est-ce qu'il faut remettre l'action dans la queue ?
                    pass
        
        # On quitte
        self.close()
        
        return self.time_counter_s
    

    def displayScore(self) :
        """Affichage du score sur l'image"""

        # Define font spec
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 0)
        thickness = 1
        
        # Define positions for the text (relative)
        position = (0.05, 0.05)
        position2 = (0.05, 0.15)

        # Largeur de la barre max en relative de l'image
        bar_max_width = 0.5

        # Define text
        text_count = f"{self.exo_name} : "
        text_tol   = f"Tolerance : "

        # Convert position : relative -> absolute
        h, w, _ = self.image.shape
        position_count = (int(round( position[0]*w)), int(round( position[1]*h)))
        position_tol   = (int(round(position2[0]*w)), int(round(position2[1]*h)))
        bar_max_width  = int(round(bar_max_width*w))
        
        # Displays texts
        cv2.putText(self.image, text_count, position_count, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(self.image,   text_tol,   position_tol, font, font_scale, color, thickness, cv2.LINE_AA)

        # Compute text field sizes
        text_count_w, text_count_h = cv2.getTextSize(text_count, font, font_scale, thickness)[0]
        text_tol_w,     text_tol_h = cv2.getTextSize(  text_tol, font, font_scale, thickness)[0]
        text_max_w = max(text_count_w, text_tol_w)

        # Barre de "chargement" (compteur)
        bar_length = int(round(bar_max_width * self.time_counter_s / self.objective_s))
        load_top_left         = (position_count[0] +    text_max_w, position_count[1] - text_count_h)
        load_max_bottom_right = ( load_top_left[0] +   bar_max_width,  load_top_left[1] + text_count_h)
        load_bottom_right     = ( load_top_left[0] +      bar_length,  load_top_left[1] + text_count_h)
        cv2.rectangle(self.image, load_top_left, load_max_bottom_right, cst.RGB_NESSY_LIGHT_GREY, -1) # Background
        cv2.rectangle(self.image, load_top_left,     load_bottom_right,     cst.RGB_NESSY_ORANGE, -1)

        # Barre de "vie" (tolérance)
        bar_length = int(round(bar_max_width * self.tol_s / self.tol_max_s))
        life_top_left         = ( position_tol[0] +    text_max_w,  position_tol[1] - text_tol_h)
        life_max_bottom_right = (life_top_left[0] + bar_max_width, life_top_left[1] + text_tol_h)
        life_bottom_right     = (life_top_left[0] +    bar_length, life_top_left[1] + text_tol_h)
        cv2.rectangle(self.image, life_top_left, life_max_bottom_right, cst.RGB_NESSY_LIGHT_GREY, -1) # Background
        cv2.rectangle(self.image, life_top_left,     life_bottom_right,     cst.RGB_NESSY_BLUE, -1)

        # Displays texts
        text_count = f" ({self.time_counter_s:.1f} s) "
        cv2.putText(self.image, text_count, load_max_bottom_right, font, font_scale, color, thickness, cv2.LINE_AA)
        

    def booo(self) :
        """Affichage du booooo !"""

        # TODO : mettre en positions relatives !

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 95, 31)
        thickness = 2
        text1 = "Booooooooooooh !"
        position1 = (200, 100)
        text2 = "you failed at"
        position2 = (220, 150)
        text3 = f"{self.time_counter_s:.1f} sec."
        position3 = (220, 225)
        text4 = f"of the {self.exo_name} !"
        position4 = (180, 275)
        cv2.putText(self.image, text1, position1, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(self.image, text2, position2, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(self.image, text3, position3, font, font_scale+1, color, thickness+1, cv2.LINE_AA)
        cv2.putText(self.image, text4, position4, font, font_scale, color, thickness, cv2.LINE_AA)


        
