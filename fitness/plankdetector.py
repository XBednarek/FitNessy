# Import relatif dans le package :
from .detector import Detector
from . import constants as cst # <-- Pour utiliser les constantes
from .tools import image2position, calcul_angle
# (Pour exécuter ce fichier, il faut donc faire proprement depuis l'extérieur du package)
# Exemple : uv run python3 -m fitness.pushupdetector
import time

# Autres imports
import numpy as np
import cv2
import joblib

class PlankDetector(Detector):

    # -------------------------------------------------------------------------
    #                                                              Constructeur
    # -------------------------------------------------------------------------

    def __init__(self, mediapipe_model, cap, verbose:bool = False, show_landmark:bool = False, windows_name:str = cst.WIN_NAME_PLANK) -> None:
        """Constructeur"""
        # Appel explicite du constructeur parent
        super().__init__(mediapipe_model, cap, verbose, show_landmark, windows_name)


    # -------------------------------------------------------------------------
    #                                                                  Méthodes
    # -------------------------------------------------------------------------
    
    # Masquage
    def run(self, objective:int) -> float:
            
            plank_active = False
            start_time = 0
            elapsed = 0
            tolerance_time = 2
            tolerance_start = None

            while self.cap.isOpened():
                 
                # Process de l'image de la webcam
                success = self.read_and_process()
            
                if not success :
                    if self.verbose :
                        print("Ignoring empty camera frame.")
                    continue

                # Récupération de l'array des positions
                positions = self.getPositions()

                if positions is None:
                    detection = False
                else:
                    detection = self.detect(positions)  

                # --------- Detect start plank and end plank ---------
                if detection:
                    if not plank_active:
                        # plank starting
                        plank_active = True
                        # start time
                        start_time = time.time()
                        if self.verbose:
                            print("Plank started!")
                    # if plank strat, tolerance is reset
                    tolerance_start = None
                else:
                    # not detection but the person was already in plank
                    if plank_active:
                        if tolerance_start is None:
                            # strat tolerance
                            tolerance_start = time.time()
                        # if tme is up to tolerence, the exercie end
                        elif time.time() - tolerance_start >= tolerance_time:
                            plank_active = False
                            # check the total time
                            elapsed = time.time() - start_time
                            if self.verbose:
                                print(f"Plank stopped! Duration: {elapsed:.2f}s")
                            # end the exercice
                            break  

                # --------- retrieve timer ---------
                if plank_active:
                    elapsed = time.time() - start_time

                # --------- Display timmer ---------
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (0, 0, 255)
                thickness = 2
                text = f"Plank: {elapsed:.1f} s"
                cv2.putText(self.image, text, (10,50), font, font_scale, color, thickness, cv2.LINE_AA)

                # Affichage
                self.imshow()

                #------------- manage when objective is reach ------
                if elapsed >= objective :
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # set up size
                    font_scale = 1
                    # set up color
                    color = (255, 95, 31)
                    # set up thikness
                    thickness = 2
                    # add text on cv2 images at coordinate given
                    cv2.putText(self.image, "Congratulations !", (200, 100), font, font_scale, color, thickness, cv2.LINE_AA)
                    cv2.putText(self.image, "you performed", (220, 150), font, font_scale, color, thickness, cv2.LINE_AA)
                    cv2.putText(self.image, f"{objective} sec.", (250, 225), font, font_scale+1, color, thickness+1, cv2.LINE_AA)
                    cv2.putText(self.image, "of plank !", (180, 275), font, font_scale, color, thickness, cv2.LINE_AA)
                    # Affichage
                    self.imshow()
                    cv2.waitKey(5000)
                    break
                # --------- Set up "q" to quite ---------
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

            # On quitte
            self.close()

            return elapsed

    
    def detect(self, positions):
        """detect if the person is doing plank exercice"""
        return self.is_elbow_ankle_aligned_y(positions) and self.check_plank_angle(positions)
            
    def is_elbow_ankle_aligned_y(self,position): 
        """Check if the elbows and ankles are approximately aligned along the vertical axis (y),
           to ensure the person is in a proper plank position.

        Args:
            position (list of Landmark): 
            A list of 33 MediaPipe Pose landmarks. Each landmark should have a `.y`.

        Returns:
            bool: 
                True if the elbows and ankles are approximately aligned vertically (within tolerance), 
                False otherwise.
    
        """

        # retriev the y coordinate
        right_elbow = position[cst.RIGHT_ELBOW][1]
        left_elbow  = position[cst.LEFT_ELBOW][1]
        #right_ankle = position[cst.RIGHT_ANKLE][1]
        #left_ankle  = position[cst.LEFT_ANKLE][1]
        right_foot_index = position[cst.RIGHT_FOOT_INDEX][1]
        left_foot_index = position[cst.LEFT_FOOT_INDEX][1]

        tolerance = 0.15
        y_values = [right_foot_index,left_foot_index, right_elbow, left_elbow]

        # chek if ankel and elbow are more or less at the same y
        return max(y_values)-min(y_values) < tolerance
    
    def check_plank_angle(self,position) :

        """Check if the shoulder-hip-knee angles are approximately straight on at least one side
            to verify a proper plank posture.

        Args:
            position (list of Landmark): 
                A list of 33 MediaPipe Pose landmarks. Each landmark should have `.x`, `.y`, 
                `.z` attributes.

        Returns:
            bool: 
                True if the plank is approximately straight on at least one side, False otherwise.

        """

        # retrieve coordinate needed
        right_knee_coordinate = position[cst.RIGHT_KNEE]
        left_knee_coordinate  = position[cst.LEFT_KNEE]
        right_hip_coordinate = position[cst.RIGHT_HIP]
        left_hip_coordinate  = position[cst.LEFT_HIP]
        right_shoulder_coordinate = position[cst.RIGHT_SHOULDER]
        left_shoulder_coordinate  = position[cst.LEFT_SHOULDER]
        right_elbow_coordinate = position[cst.RIGHT_ELBOW]
        left_elbow_coordinate  = position[cst.LEFT_ELBOW]
        

        # calcul angle  shoulder hip knee
        angle_right = calcul_angle(right_elbow_coordinate,right_shoulder_coordinate,right_hip_coordinate)
        angle_left = calcul_angle (left_elbow_coordinate,left_shoulder_coordinate,left_hip_coordinate)

        # tolerance around 180°
        return 80< angle_right < 100 or 80< angle_left < 100




    






        