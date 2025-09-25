# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.movedetector

# Autres imports
import numpy as np
import cv2

class MoveDetector(Detector):
    
    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False,
                                             show_landmark:bool = False,
                                             windows_name:str = cst.WIN_NAME_DEFAULT,
                                             reward_string:str = "",
                                             movement_cycle:list[str]=[""],
                                             exo_name: str="") -> None:
        """Constructeur d'un détecteur de mouvement"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose=verbose,
                                               show_landmark=show_landmark,
                                               windows_name=windows_name,
                                               reward_string=reward_string)

        # Liste de positions attendues pour faire un movement
        movement_cycle = movement_cycle.copy()
        # Check que la liste n'est pas vide :
        if len(movement_cycle) == 0:
            raise ValueError("Il faut fournir la liste de positions qui composent le mouvement !")
        self.movement_cycle = movement_cycle

        # Compteur :
        self.counter = 0

        # Où est-ce qu'on doit être dans le cycle
        self.expected_position_in_cycle = 0

        # Nom de l'exercice :
        self.exo_name = exo_name

        # Pose détectée :
        self.detected_pose = "None"
    
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    # Implémentation de la méthode abstraite de la classe mère Detector
    def run(self, objective:int) -> float:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        
        # On suppose qu'on n'est pas encore rentrée dans le cycle du mouvement
        in_cycle = False
        
        while self.cap.isOpened():

            # Process de l'image de la webcam
            success = self.read_and_process()
        
            if not success :
                if self.verbose :
                    print("Ignoring empty camera frame.")
                continue

            # Récupération de l'array des positions
            positions = self.getPositions()
            
            if positions is not None :

                # Détection :
                self.detected_pose = self.detect(positions)

                # Affichage
                if self.verbose :
                    print(f"Pose détectée : {self.detected_pose}")

                # Si on detecte la position attendue
                if self.detected_pose == self.movement_cycle[self.expected_position_in_cycle] :

                    # On mets à jour le compteur ! (sauf si c'est la toute première détection)
                    if in_cycle :
                        self.counter += 1 / len(self.movement_cycle)
                    else :
                        in_cycle = True

                    # On doit aussi mettre à jour la prochaine position attendue

                    # Si on est à la fin, on revient au debut
                    if self.expected_position_in_cycle == (len(self.movement_cycle) -1):
                        self.expected_position_in_cycle = 0
                    # Sinon, on enchaine
                    else :
                        self.expected_position_in_cycle += 1 
            
            # Monitoring du score ou écran de féliciation
            if self.counter >= objective :
                self.congrate(objective)
            else :
                self.displayScore()

            # Affichage
            self.imshow()
            
            # Quitte si on appuie sur q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # Quitte si l'objectif est atteint
            if self.counter >= objective :
                cv2.waitKey( cst.CONGRATS_DURATION_S * 1000)
                break
        
        # On quitte
        self.close()
        
        return self.counter
    
    def displayScore(self) :
        """Affichage du score sur l'image"""

        # Define font spec
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 0)
        thickness = 1
        
        # Define positions for the text
        position = (10, 20)
        position2 = (10, 40) 
        position3 = (10, 60)

        # Define text
        text_count      = f"{self.exo_name} #: {self.counter}"
        text_status     = f"Next position expected : {self.movement_cycle[self.expected_position_in_cycle]}"
        text_lastdetect = f"Last position detected : {self.detected_pose}"

        # Displays
        cv2.putText(self.image, text_count, position, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(self.image, text_status, position2, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(self.image, text_lastdetect, position3, font, font_scale, color, thickness, cv2.LINE_AA)
        

        
