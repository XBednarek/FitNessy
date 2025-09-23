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



def image2position(image : np.ndarray, mediapipe_model, show:bool=False) -> np.ndarray | None:
    """Return pose landmarks of the image as numpy array.

    Args:
        image (np.ndarray): RGB image 
        mediapipe_model (_type_, optional): Media pipe model used. Defaults to MEDIAPIPE_BASE_MODEL.
        show (bool, optional): If we want to show the image. Defaults to False.

    Returns:
        np.ndarray: Numpy array with x, y, z as column. Or None if no position have been detected.
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
    
if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour le module tools *")
    print("-"*80)