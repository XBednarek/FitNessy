# Import relatif dans le package :

# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.button

# Autres imports
import cv2
import numpy as np

class Rectangle :

    """
    """
    
    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, img:np.ndarray, *, A:tuple[float, float],
                                          B:tuple[float, float],
                                          line_color:tuple[int, int, int] = (0, 0, 0),
                                          line_thickness: float = 1.,
                                          text:str = '',
                                          text_color:tuple[int, int, int] = (0, 0, 0),
                                          text_font = cv2.FONT_HERSHEY_SIMPLEX,
                                          text_thickness:float = 1.,
                                          text_font_scale:float = 1.) -> None:
        """Constructeur d'un rectangle avec un texte dedans.

        Args:
            img (np.ndarray): image cv2
            A (tuple[float, float]): Point en haut à gauche du rectangle (en coord. relative de l'image)
            B (tuple[float, float]): Point en bas à droite du rectangle (en coord. relative de l'image)
            line_color (tuple[int, int, int], optional): Couleur du contour. Defaults to (0, 0, 0).
            line_thickness (float, optional): Epaisseur du contour. Defaults to 1..
            text (str, optional): Texte dans le rectangle. Defaults to ''.
            text_color (tuple[int, int, int], optional): Couleur du texte. Defaults to (0, 0, 0).
            text_font (_type_, optional): Font du texte. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
            text_thickness (float, optional): Epaisseur du texte. Defaults to 1..
            text_font_scale (float, optional): Echelle du texte. Defaults to 1..

        Raises:
            ValueError: Si les coordonnées de A et B ne sont pas entre 0 et 1

        Un petit schéma :

              x       
           O----->
           |           A ----------*      A = (x_A, y_A)
         y |           |           |      B = (x_B, y_B)
           v           |           |      avec x en horizontal  et y en verticale,
                       *-----------B      le point (0, 0) étant en haut à gauche

        """

        # Check que A et B sont en relatif
        if A[0] < 0. or A[0] > 1. or A[1] < 0. or A[1] > 1. or B[0] < 0. or B[0] > 1. or B[1] < 0. or B[1] > 1. :
            raise ValueError("A et B doivent être en coordonnées relatives ! (entre 0 et 1 donc)")

        # Conversion des coordonnées relatives en coordonnées absolues (pixels)
        h, w, _ = img.shape
        A = (int(round(A[0]*w)), int(round(A[1]*h)))
        B = (int(round(B[0]*w)), int(round(B[1]*h)))

        # Ajouts du rectangle sur l'image
        cv2.rectangle(img, A, B, line_color, line_thickness)
  
        # Calcul de la position du texte qui doit être au milieu du rectangle
        text_width, text_height = cv2.getTextSize(text, text_font, text_font_scale, text_thickness)[0]
        text_position = (int( (B[0]+A[0]) / 2) - int(text_width / 2), int((B[1]+A[1]) / 2) + int(text_height / 2))
        
        # Ajout du texte sur l'image
        cv2.putText(img, text, text_position, text_font, text_font_scale, text_color, text_thickness, cv2.LINE_AA)

        # Ajout des membres
        self.x_min = A[0]
        self.x_max = B[0]
        self.y_min = A[1]
        self.y_max = B[1]

    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------

    def contains(self, x, y) :
        """Vérifie si le point x, y est dans le rectangle"""
        return (x >= self.x_min and x <= self.x_max and y >= self.y_min and y <= self.y_max )
      