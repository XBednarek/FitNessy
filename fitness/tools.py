""" Outils utiles pour l'application de Fitness """

# Import relatif dans le package :

# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.tools

# Autres imports
import numpy as np
import cv2
from .given_tools import draw_holistic_results 

def landmark2array(pose_landmarks) -> np.ndarray :
    """Convert pose_landmarks to numpy array

    Args:
        pose_landmarks : pose_landmarks as given by results.pose_landmarks

    Returns:
        np.ndarray: numpy array shape = (33, 3)
    """
    return np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks.landmark], dtype=np.float32)

def landmark2array_visibility(pose_landmarks) -> np.ndarray :
    """Extraire la visibility


    Args:
        pose_landmarks : pose_landmarks as given by results.pose_landmarks

    Returns:
        np.ndarray: numpy array shape = (33, 3)
    """
    return np.array([[landmark.visibility] for landmark in pose_landmarks.landmark], dtype=np.float32)

def get_points_visibles(positions:np.ndarray, visibilities:np.ndarray, points_indices :np.ndarray, image_shape, seuil: float = 0.5) :
        """
        Vérifie si un landmark est visible au-dessus d'un seuil.
        Args:
            points_indices (int): index du point (0-32)
            images_shape: hauteur et longeur
            seuil (float): seuil de visibilité (0-1)
        Returns:
            points_visible: x_pixels_visible,y_pixels_visible
        """

        h,w, _ = image_shape

        points_visible={}

        # normaliser l'image et detecter quel coté est plus visible que l'autre
        for  name,idx in points_indices.items():
            vis=visibilities[idx,0]
            if vis > seuil:
                x,y = positions[idx,0],positions[idx,1]
                points_visible[name]= (int(x * w), int(y * h))

        return points_visible


def image2position(image : np.ndarray, mediapipe_model, show:bool=False) -> np.ndarray | None:
    """Return pose landmarks of the image as numpy array.

    Args:
        image (np.ndarray): RGB image 
        mediapipe_model (_type_, optional): Media pipe model used. Defaults to MEDIAPIPE_BASE_MODEL.
        show (bool, optional): If we want to show the image. Defaults to False.

    Returns:
        np.ndarray: Numpy array with x, y, z as column. Or None if no position have been detected.
        object : results of the process
    """

    # Traitement
    results = mediapipe_model.process(image)

    # Récupération des landmarks
    if (results is not None) and (results.pose_landmarks is not None) :
        pose_landmarks = results.pose_landmarks

        # Affichage si nécessaire
        if show :
            result_image = draw_holistic_results(image, results, show_face=False, show_hands=False, show_pose=True)
            key_close = 'q'
            window_title = "MediaPipe ('"+ key_close +"'pour quitter)"
            result_image =  cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_title, result_image)
            key = cv2.waitKey(10)

            while key != ord(key_close) :
                key = cv2.waitKey(10) & 0xFF # Bit masking (comme précisé en dessous)
            cv2.destroyWindow(window_title)
            cv2.waitKey(1)  # On attend encore un peu au cas où

        # Conversion vers un array numpy  
        landmarks_array = landmark2array(pose_landmarks)

        return landmarks_array
    else :
        return None
    

def calc_distance(array, point1, point2):
    """
    Calculate the Euclidean distance between two landmarks
    """
    # point 1
    point1_y = array[point1][1]
    point1_x = array[point1][0]
    # point 2
    point2_y = array[point2][1]
    point2_x = array[point2][0]

    # y distance
    if point1_y > point2_y:
        dist_y = point1_y - point2_y
    else:
        dist_y = point2_y - point1_y
    # x distance
    if point1_x > point2_x:
        dist_x = point1_x - point2_x
    else:
        dist_x = point2_x - point1_x

    dist = (dist_x**2 + dist_y**2)**0.5

    return dist


def calcul_angle(A,B,C):
    """
    calculer langle entre le genou , cheville et hanche 
    """
    A=np.array(A)
    B=np.array(B)
    C=np.array(C)

    # calculer les vecteurs AB et CB ( A: epaule, B: coude, C:Poinet )
    AB=A-B
    CB=C-B

    # cacluler l'angle 

    cos_angle=np.dot(AB,CB)/(np.linalg.norm(AB) * np.linalg.norm(CB))

    # Convertir en degrés
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle



def to_float32(image: np.ndarray) -> np.ndarray:
    """Convert to float32 image"""

    # Copy :
    image = image.copy()

    # Convert if needed
    if image.dtype == np.float32 :
        return image
    if image.dtype == np.float64 :
        return image.astype(np.float32)
    elif image.dtype == np.uint8 :
        return np.clip(image/255, 0, 1).astype(np.float32)
    else :
        raise ValueError(f"I can not convert {image.dtype} image to float32 image !")
    


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert to uint8 image"""

    # Copy :
    image = image.copy()

    # Convert if needed
    if image.dtype == np.uint8 :
        return image
    elif (image.dtype == np.float32) or (image.dtype == np.float64):
        return np.clip(image*255, 0, 255).astype(np.uint8)
    else :
        raise ValueError(f"I can not convert {image.dtype} image to float32 image !")



def composite_image(imageA: np.ndarray, imageB: np.ndarray, mask: np.ndarray, dtype:type=np.float32) -> np.ndarray:
    """ Blend two images using an alpha mask.
        Return an image in the given dtype

    Arguments:
        imageA (np.ndarray): Background image (could be uint8 or float32, could be RGB or RGBA)
        imageB (np.ndarray): Foreground image (could be uint8 or float32, could be RGB or RGBA)
        mask (np.ndarray): Alpha mask with values in range [0, 1]
        
    Returns:
        np.ndarray: The blended image (RGB) in the given dtype
    """

    # On va arbitrairement travailler sur des float32
    imageA = to_float32(imageA)
    imageB = to_float32(imageB)

    # On fait le alpha blending
    try :
        imageC = (imageB * mask + imageA * (1-mask))
    except :
        imageC = (imageB * mask[..., None] + imageA * (1-mask[..., None] ))

    # On convertit selon le dtype souhaité
    if dtype == np.uint8 :
        return to_uint8(imageC)
    else :
        return imageC
    
def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Redimensionner une image en lui donnant un nouvelle hauteur"""
    # Obtenir les dimensions actuelles
    h, w = image.shape[:2]
    
    # Calculer la nouvelle largeur pour garder les proportions
    new_width = int(target_height / h * w)
    
    # Redimensionner
    resized_image = cv2.resize(image, (new_width, target_height))

    return resized_image


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Redimensionner une image en lui donnant un nouvelle largeur"""
    # Obtenir les dimensions actuelles
    h, w = image.shape[:2]
    
    # Calculer la nouvelle hauteur pour garder les proportions
    new_height = int(target_width / w * h)
    
    # Redimensionner
    resized_image = cv2.resize(image, (target_width, new_height))
    
    return resized_image


if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour le module tools *")
    print("-"*80)