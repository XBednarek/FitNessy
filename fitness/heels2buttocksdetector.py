# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.heels2buttocksdetector

# Autres imports
import numpy as np

class Heels2ButtocksDetector(Detector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose)


    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    
    # Masquage
    def run(self, objective:int) -> int:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        return 5

    # Masquage
    def detect(self, positions: np.ndarray) -> str:
        """Détecte une position"""
        # TODO
        pass

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour la classe Heels2ButtocksDetector *")
    print("-"*80)
