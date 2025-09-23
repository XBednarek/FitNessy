""" Outils donnés dans le notebook ACV_Part1_Project.ipynb.
    Avec quelques modifications mineures"""

# Import relatif dans le package :

# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.given_tools

# Autres imports
import cv2
import mediapipe as mp

drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

def run_filter_with_mediapipe_model(mediapipe_model, mediapipe_based_filter=None, endfilter=None):
    """Run a media pipe model on each video frame grabbed by the webcam and draw results on it

    Args:
        mediapipe_model (): A mediapipe model
        mediapipe_based_filter (): a function to draw model results on frame

    Returns:
        np.ndarray, mediapipe model result
    """
    cap = cv2.VideoCapture(0)
    
    try:
        with mediapipe_model as model:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue     # If loading a video, use 'break' instead of 'continue'.

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                if mediapipe_based_filter is not None :
                    try:
                        results = model.process(image)
                    except Exception:
                        results = None
                else :
                    results = None

                if mediapipe_based_filter is not None :
                    if results and results.pose_landmarks:
                        result_image = mediapipe_based_filter(image, results)
                    else:
                        result_image = image
                else :
                    result_image = image

                # [XB] j'ai rajouté la possibilité d'avoir un filtre simple !
                if endfilter is not None :
                    result_image = endfilter(result_image)
                    

                # Je mets plutôt la conversion ici, mais ducoup ca peut changer des choses dans le display
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

                cv2.imshow('MediaPipe', result_image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return image, results


def draw_holistic_results(image, results, show_hands=True, show_face=True, show_pose=True):
    if show_hands:
        drawing_utils.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
        )

        drawing_utils.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
        )

    if show_face:
        drawing_utils.draw_landmarks(
            image,
            results.face_landmarks,
            mp.solutions.holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(thickness=0, circle_radius=0, color=(255, 255, 255)),
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )

    if show_pose:
        drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
        )
    
    return image

if __name__=='__main__':
    # Tests
    print("-"*80)
    print(" * Tests pour le module given_tools *")
    print("-"*80)


    Holistic = mp.solutions.holistic.Holistic

    # Global variables to keep state between frames
    pushup_count = 0
    pushup_status = "up"

    last_image, last_results = run_filter_with_mediapipe_model(
        mediapipe_model=Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5),
        mediapipe_based_filter=draw_holistic_results
        )

    print(f"{last_image.shape = }")