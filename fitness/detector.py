import numpy as np

class Detector :
    """
    """
    
    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False) -> None:
        """"""
        # Modèle mediapipe
        self.mediapipe_model = mediapipe_model
        # Capture vidéo :
        self.cap = cap
        # Réglage de la verbosité
        self.verbose = verbose
    
    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    
    # Abstraite !
    def run(self, objective:int) -> int:
        """Run le décompte et renvoie le nombre de fois que l'exercice à été
           réalisé"""
        raise NotImplementedError("Cette fonction doit être implémentée dans la classe fille !")
        
    # Abstraite !
    def detect(self, positions: np.ndarray) -> str:
        """Détecte une position"""
        raise NotImplementedError("Cette fonction doit être implémentée dans la classe fille !")

