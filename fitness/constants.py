# Ca serait bien de mettre ici toutes les constantes du projets,
# par exemple les positions des points qui sont utiles
import mediapipe as mp

# POSITION DU CORPS : 

# Coté droit
RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
RIGHT_ELBOW = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
RIGHT_WRIST = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
RIGHT_HIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP

# Coté gauche
LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
LEFT_WRIST = mp.solutions.pose.PoseLandmark.LEFT_WRIST
LEFT_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP


# Nom des exercices
EX_PUSH_UP = "Pompes"
EX_HELLS2BUTTOCKS = "Talons-fesses"
EX_KNEERAISE = "Montées de genoux"
EX_SQUATS = "Squats"

# MOUVEMENTS 
MOVE_UP = "move up"
MOVE_DOWN = "move down"
MOVE_UNKNWON = "unknown"
MOVE_LEFT_UP_RIGHT_DOWN = "left up right down"
MOVE_LEFT_DOWN_RIGHT_UP = "left down right up"
MOVE_LEFT_DOWN_RIGHT_DOWN = "left down right down"