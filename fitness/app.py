# Import relatif dans le package :
from .heels2buttocksdetector import Heels2ButtocksDetector
from .kneeraisedetector import KneeRaiseDetector
from .pushupdetector import PushUpDetector
from .squatdetector import SquatDetector
from .rectangle import Rectangle
from .tools import composite_image, resize_to_height, to_float32, to_uint8
from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.app

# Autres imports
import numpy as np
import cv2
import mediapipe as mp
from screeninfo import get_monitors

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
        # Est-ce qu'on veut tracer les landmarks ?
        self.show_landmarks = False

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

        # Setup de la résolution de l'écran ;
        self.screen_height = None
        self.screen_width = None
        self.screen_scale = 1.
        self.setScreenSize(h=cst.SCREEN_H, w=cst.SCREEN_W)


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

        # Chargement du logo :
        logo = cv2.imread('fitness/logo/logo.png', cv2.IMREAD_UNCHANGED) # Pour charger aussi le alpha
        # Reduction du logo à la bonne taille
        logo = resize_to_height(logo, int(self.screen_height*0.7))

        # Détermination du padding (pour centrer le logo)
        top_pad = int((self.screen_height-logo.shape[0])/2)
        bottom_pad = self.screen_height - logo.shape[0] - top_pad
        left_pad = int((self.screen_width-logo.shape[1])/2)
        right_pad = self.screen_width - logo.shape[1] - left_pad

        # On met l'image au bon format
        logo = np.pad(logo, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), 'constant', constant_values=0)
        logo = to_float32(logo)

        # Création de l'écran d'acceuil :
        # 1 - Page blanche : 
        screen = np.full((self.screen_height, self.screen_width, 3), 1., dtype = np.float32)
        # 2 - Placement du logo au milieu
        screen = composite_image(screen[:,:,:3], logo[:,:,:3], logo[:,:,3])
        print(f"{screen.shape =}")
        screen = to_uint8(screen)
        # 3 - Rectangles pour cliquer :
        rect_go = Rectangle(screen, A=(0.37, 0.8), B=(0.46, 0.9), text='GO !',
                                      line_color=cst.BGR_NESSY_ORANGE, line_thickness=2,
                                      text_color=cst.BGR_NESSY_ORANGE, text_font=cv2.FONT_HERSHEY_SIMPLEX,
                                      text_thickness=2, text_font_scale=1*self.screen_scale)
        rect_quit = Rectangle(screen, A=(0.54, 0.8), B=(0.63, 0.9), text='Quit',
                                      line_color=cst.BGR_NESSY_BLUE, line_thickness=2,
                                      text_color=cst.BGR_NESSY_BLUE, text_font=cv2.FONT_HERSHEY_SIMPLEX,
                                      text_thickness=2, text_font_scale=1*self.screen_scale)
        
        while self.cap.isOpened(): 

            current_screen = screen.copy()

            # Bouton option : show landmarks 
            if self.show_landmarks :
                color = cst.BGR_NESSY_ORANGE
            else :
                color = cst.BGR_NESSY_LIGHT_GREY

            rect_shldmrks = Rectangle(current_screen, A=(0.4, 0.15), B=(0.6, 0.2), text='show landmark',
                            line_color=color, line_thickness=1,
                            text_color=color, text_font=cv2.FONT_HERSHEY_SIMPLEX,
                            text_thickness=1, text_font_scale=0.5*self.screen_scale)
            
            # Affichage
            cv2.imshow(win, current_screen)


            # Capture du clic
            if rect_go.contains(self.left_clicked_x, self.left_clicked_y):
                go = True
                break
            
            if rect_quit.contains(self.left_clicked_x, self.left_clicked_y):
                break

            if rect_shldmrks.contains(self.left_clicked_x, self.left_clicked_y):
                self.show_landmarks = not self.show_landmarks
                if self.verbose :
                    print(f"{self.show_landmarks =} ")
                self.left_clicked_x = -1
                self.left_clicked_y = -1
            
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

    def setScreenSize(self, *, w:int|None = None, h:int|None = None):
        """Règle la taille de l'image"""
        # Si la taille n'est pas donnée, on met celle de l'écran
        if (w is None) or (h is None) :
            w, h = App.getScreenSize()

        # On fait bien sur de même pour la webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Mais il faut vérifier finalement que c'est OK avec la webcam
        self.screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # On règle aussi le scale par rapport à quand on dev :
        self.screen_scale = self.screen_height/cst.SCREEN_H_DEV

        # Un peu de verbosité
        if self.verbose :
            if (w == self.screen_width) and (h == self.screen_height) :
                print(f"Résolution d'écran réglée sur : wxh = {self.screen_width}x{self.screen_height}")
            else :
                print(f"On a tenté de régler la résolution d'écran sur : wxh = {w}x{h}")
                print(f"Mais finalement pour que ce soit OK pour la webcam on l'a mise sur : wxh = {self.screen_width}x{self.screen_height}")
            print(f"{self.screen_scale = }")

    @staticmethod
    def getScreenSize() -> tuple[int, int] :
        """ Pour obtenir la résolution de l'écran principal

        Returns:
            tuple[int, int]: (largeur, hauteur)
        """
        monitors = get_monitors()
        screen_width = monitors[0].width
        screen_height = monitors[0].height
        return (screen_width, screen_height)
        

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
            cst.EX_PUSH_UP: 7}
    
    # Run de l'écran test :
    go = app.show_start_screen()

    
    # Run du set :
    print("-"*10)
    print(" * Run d'exercices *")
    if go :
        app.run_exercice_session(exos)

    
