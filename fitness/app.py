# Import relatif dans le package :
from .heels2buttocksdetector import Heels2ButtocksDetector
from .kneeraisedetector import KneeRaiseDetector
from .pushupdetector import PushUpDetector
from .squatdetector import SquatDetector
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
        go  = False

        # Chargement de l'écran de démarrage
        screen = cv2.imread('fitness/logo/screen_1.png')

        # Zone de textes cliquables
        button_go = {'rect_start': (295, 400),
                     'rect_end': (345, 450),
                     "line_color": (0, 255, 0),
                     "line_thickness": 2,
                     'text': 'GO !',
                     "text_color": (0, 255, 0),
                     "text_font": cv2.FONT_HERSHEY_SIMPLEX,
                     "text_thickness": 1,
                     "text_font_scale": 0.5}

        # Ajouts des zones cliquables
        cv2.rectangle(screen, button_go['rect_start'], button_go['rect_end'], button_go['line_color'], button_go['line_thickness'])
        text_position = ( int(0.8 * button_go['rect_start'][0] + 0.2 * button_go['rect_end'][0]),
                          int(0.8 * button_go['rect_start'][1] + 0.2 * button_go['rect_end'][1]))
        cv2.putText(screen, button_go['text'], text_position, button_go['text_font'], button_go['text_font_scale'], button_go['text_color'], button_go['text_thickness'], cv2.LINE_AA)

        # Gestion du callback de la souris
        win = "First screen"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.mouse_callback)

        while  self.cap.isOpened(): 

            # Affichage
            cv2.imshow(win, screen)

            # Capture du clic
            if (self.left_clicked_x >= button_go['rect_start'][0] and 
                self.left_clicked_y >= button_go['rect_start'][1] and
                self.left_clicked_x <= button_go['rect_end'][0] and 
                self.left_clicked_y <= button_go['rect_end'][1] ) :
                if self.verbose :
                    print(button_go['text'])
                go = True
                break
            
            # Quitter au clavier
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

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
            if exo == cst.EX_HELLS2BUTTOCKS :
                score = self.heels_2_buttocks_detector.run(objectif)
                scores[exo] = score
            elif exo == cst.EX_PUSH_UP :
                score = self.push_up_detector.run(objectif)
                scores[exo] = score
            elif exo ==  cst.EX_SQUATS:
                score=self.squats_detector.run(objectif)
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
            cst.EX_HELLS2BUTTOCKS: 5,
            cst.EX_SQUATS: 5,
            cst.EX_PUSH_UP: 7}
    
    # Run de l'écran test :
    go = app.show_start_screen()
    
    # Run du set :
    print("-"*10)
    print(" * Run d'exercices *")
    if go :
        app.run_exercice_session(exos)

    
