# Import relatif dans le package :
from .heels2buttocksdetector import Heels2ButtocksDetector
from .kneeraisedetector import KneeRaiseDetector
from .pushupdetector import PushUpDetector
from .squatdetector import SquatDetector
from .bhujangasandetector import Bhujangasandetector
from .rectangle import Rectangle
from .plankdetector import PlankDetector

from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.app

# Autres imports
import numpy as np
import cv2
import mediapipe as mp

class App :
    """
    """
    
    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, *, verbose:bool = False) -> None:
        """Constructeur de l'application"""
        
        if verbose :
            print("Création de l'application")
        
        # Modèle mediapipe :
        self.mediapipe_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Capture vidéo :
        self.cap = cv2.VideoCapture(0)
        # Réglage de la verbosité
        self.verbose = verbose

        # Construction des detecteurs :
        # Pompes
        self.push_up_detector = PushUpDetector(self.mediapipe_model, self.cap, self.verbose)
        # Talons-fesses
        self.heels_2_buttocks_detector = Heels2ButtocksDetector(self.mediapipe_model, self.cap, self.verbose)
        # Squats
        self.squats_detector = SquatDetector(self.mediapipe_model, self.cap, self.verbose)
        # Montée de genou
        self.knee_raise_detector = KneeRaiseDetector(self.mediapipe_model, self.cap, self.verbose)
        #cobra yoga
        self.cobra_detector=Bhujangasandetector(self.mediapipe_model, self.cap, self.verbose)
        # planche
        self.plank_detector = PlankDetector(self.mediapipe_model,self.cap,self.verbose)

        # Positions de cliques de la souris
        self.left_clicked_x = -1
        self.left_clicked_y = -1
        self.right_clicked_x = -1
        self.right_clicked_y = -1


    def __del__(self):
        """Destructeur de l'application"""
        if self.verbose :
            print("Destruction de l'application")
        try:
            if hasattr(self, 'mediapipe_model') and self.mediapipe_model is not None:
                # self.mediapipe_model.close() --> problème avec ca
                self.mediapipe_model = None
            
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()

            if cv2 is not None:
                cv2.destroyAllWindows()
    
        except:
            print("Erreur lors de la destruction")
            pass  # Ignore les erreurs lors de la destruction

    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    
    def show_start_screen(self) -> bool :
        """Affichage de l'écran de démarrage de l'application"""

        # Variable de sortie :
        go = False

        # Gestion du callback de la souris
        win = cst.WIN_NAME_TITLE
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.mouse_callback)

        # Chargement de l'écran de démarrage
        screen = cv2.imread('fitness/logo/screen_1.png')

        # Rectangles pour cliquer :
        rect_go = Rectangle(screen, A=(0.35, 0.8), B=(0.45, 0.94), text='GO !',
                                      line_color=cst.BGR_NESSY_BLUE, line_thickness=2,
                                      text_color=cst.BGR_NESSY_ORANGE, text_font=cv2.FONT_HERSHEY_SIMPLEX,
                                      text_thickness=1, text_font_scale=0.5)
        
        rect_quit = Rectangle(screen, A=(0.55, 0.8), B=(0.65, 0.94), text='Quit !',
                                      line_color=cst.BGR_NESSY_BLUE, line_thickness=2,
                                      text_color=cst.BGR_NESSY_ORANGE, text_font=cv2.FONT_HERSHEY_SIMPLEX,
                                      text_thickness=1, text_font_scale=0.5)

        while self.cap.isOpened(): 

            # Affichage
            cv2.imshow(win, screen)

            # Capture du clic
            if rect_go.contains(self.left_clicked_x, self.left_clicked_y):
                go = True
                break
            
            if rect_quit.contains(self.left_clicked_x, self.left_clicked_y):
                break
            
            # Quitter au clavier
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Destruction de la fenetre
        cv2.destroyWindow(win)

        return go

    # Lancement d'un set d'exercice
    def run_exercice_session(self, exercices:dict) :
        """ Lancement d'un session d'exercice

        Args:
            exercices (dict): Dictionnaire de la séance
        """

        # Dictionnaire des scores :
        scores = {}
        # Lancement de la séance d'exercice
        for exo, objectif in exercices.items():
            if self.verbose:
                print(f"Objectif {objectif:d} {exo}.")
            if exo == cst.EX_HEELS2BUTTOCKS :
                score = self.heels_2_buttocks_detector.run(objectif)
                scores[exo] = score
            elif exo == cst.EX_PUSH_UP :
                score = self.push_up_detector.run(objectif)
                scores[exo] = score
            elif exo ==  cst.EX_SQUATS:
                 score=self.squats_detector.run(objectif)
                 scores[exo] = score
            elif exo ==  cst.EX_KNEERAISE:
                score=self.knee_raise_detector.run(objectif)
                scores[exo] = score
            elif exo ==  cst.EX_PLANK :
                score = self.plank_detector.run(objectif)
                scores[exo] = score
            elif exo ==  cst.EX_COBRA:
                score=self.cobra_detector.run(objectif)
                scores[exo] = score

        # Résultats de la séance :
        for exo, score in scores.items():
            print(f"Score : {score:.1f} {exo}.")

    # -------------------------------------------------------------------------
    #                                                 Méthodes pour l'affichage
    # -------------------------------------------------------------------------

    def mouse_callback(self, event, x, y, flags, param):
        """Définition du callback pour le clique souris"""

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.verbose :
                print(f"Clic gauche à ({x}, {y})")
            self.left_clicked_x = x
            self.left_clicked_y = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.verbose :
                print(f"Clic droit à ({x}, {y})")
            self.right_clicked_x = x
            self.right_clicked_y = y
        

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe App *")
    print("-"*80)

    # Création :
    print("-"*10)
    print(" * Création *")
    app = App(verbose=True)

    # Set d'exercice :

    exos = {cst.EX_KNEERAISE: 10,
             cst.EX_HEELS2BUTTOCKS: 5,
             cst.EX_SQUATS: 5,
             cst.EX_PUSH_UP: 7,
             cst.EX_COBRA: 10}

    
    # Run de l'écran test :
    go = app.show_start_screen()

    
    # Run du set :
    print("-"*10)
    print(" * Run d'exercices *")
    if go :
        app.run_exercice_session(exos)

    