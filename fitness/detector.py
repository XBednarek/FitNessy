# Import relatif dans le package :
from .tools import landmark2array, landmark2array_visibility
from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.detector

# Autres imports
import numpy as np
import cv2
import mediapipe as mp

class Detector :
    """
    """
    
    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False,
                                             show_landmark:bool = False,
                                             windows_name:str = cst.WIN_NAME_DEFAULT,
                                             reward_string:str = "") -> None:
        """"""
        # Modèle mediapipe
        self.mediapipe_model = mediapipe_model
        # Capture vidéo :
        self.cap = cap
        # Réglage de la verbosité
        self.verbose = verbose
        # Est-ce qu'on affiche les landmarks ?
        self.show_landmarks = show_landmark
        # On retient la frame courante (ie la dernière prise par la camera)
        self.frame = None # BGR
        # Et le résultat courant du process mediapipe
        self.result = None
        # Mais aussi l'image qu'on affiche
        self.image = None # RGB
        # Est-ce qu'on veut inverser l'image ?
        self.flip_frame = False
        # Nom de la fenètre d'affichage
        self.windows_name = windows_name
        # String pour le reward
        self.reward_string = reward_string
    
    # -------------------------------------------------------------------------
    #                                                       Méthodes Abstraites
    # -------------------------------------------------------------------------
    
    # Abstraite !
    def run(self, objective:int) -> float:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        raise NotImplementedError("Cette fonction doit être implémentée dans la classe fille !")
        
    # Abstraite !
    def detect(self, positions: np.ndarray, visibility: np.ndarray) -> str:
        """Détecte une position"""
        raise NotImplementedError("Cette fonction doit être implémentée dans la classe fille !")
    
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    def read_and_process(self) -> bool:
        """Lit l'image de la webcam et fait le process mediapipe"""
        
        if self.cap.isOpened():

            # Lecture
            success, self.frame = self.cap.read()

            if not success:
                return success

            # Est-ce qu'on est en mode mirroir ?
            if self.flip_frame :
                self.frame = cv2.flip(self.frame, 1)

            # Process mediapipe
            self.results = self.mediapipe_model.process(self.frame)

            # Creation de l'image à afficher
            self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # Affichage des landmarks si necessaire
            if self.show_landmarks :
                mp.solutions.drawing_utils.draw_landmarks(
                    self.image,
                    self.results.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )

            # Retour :
            return success
        else :
            return False

    def getPositions(self) -> np.ndarray | None :
        """Renvoie l'array numpy des positions détectées"""

        if (self.results is not None) and (self.results.pose_landmarks is not None) :
            return landmark2array(self.results.pose_landmarks)
        else :
            return None

    def getVisibility(self) -> np.ndarray | None :
        """Renvoie l'array numpy des visibility détectées"""

        if (self.results is not None) and (self.results.pose_landmarks is not None) :
            return landmark2array_visibility(self.results.pose_landmarks)
        else :
            return None

    def imshow(self):
        """Affichage de l'image"""
        cv2.imshow(self.windows_name, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

    def close(self):
        """Fermeture de la fenètre"""
        cv2.destroyWindow(self.windows_name)

    def congrate(self, objective):
        """Génère l'écran de félicitation""" 

        # TODO : faire en relatif de la taille de l'image !

        font = cv2.FONT_HERSHEY_SIMPLEX
        # set up size
        font_scale = 1
        # set up color
        color = (255, 95, 31)
        # set up thikness
        thickness = 2

        # outline
        
        # outline color
        outline_color = (0, 0, 0)  # noir pour le contour
        outline_thickness = thickness + 2  # contour légèrement plus épais

        # --------- congrats message avec contour ------------

        # "Congratulations !"
        cv2.putText(self.image, "Congratulations !", (200, 100), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(self.image, "Congratulations !", (200, 100), font, font_scale, color, thickness, cv2.LINE_AA)

        # "you performed"
        cv2.putText(self.image, "you performed", (220, 150), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(self.image, "you performed", (220, 150), font, font_scale, color, thickness, cv2.LINE_AA)

        # "{objective} sec."
        cv2.putText(self.image, f"{objective}", (250, 225), font, font_scale+1, outline_color, outline_thickness+1, cv2.LINE_AA)
        cv2.putText(self.image, f"{objective}", (250, 225), font, font_scale+1, color, thickness+1, cv2.LINE_AA)

        # "of plank !"
        cv2.putText(self.image, self.reward_string, (220, 275), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(self.image, self.reward_string, (220, 275), font, font_scale, color, thickness, cv2.LINE_AA)

        # Confettis
        for _ in range(200):  
            x = np.random.randint(0, self.image.shape[1])
            y = np.random.randint(0, self.image.shape[0])
            col = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            cv2.circle(self.image, (x, y), 3, col, -1)
            

         


    

