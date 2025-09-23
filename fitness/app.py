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
    # exos = {cst.EX_KNEERAISE: 10,
    #         cst.EX_HELLS2BUTTOCKS: 5,
    #         cst.EX_SQUATS: 5,
    #         cst.EX_PUSH_UP: 7}
    exos = {cst.EX_SQUATS: 3}
    
    # Run du set :
    print("-"*10)
    print(" * Run d'exercices *")
    app.run_exercice_session(exos)

    print("--")
