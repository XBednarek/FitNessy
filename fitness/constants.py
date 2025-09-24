# Ca serait bien de mettre ici toutes les constantes du projets,
# par exemple les positions des points qui sont utiles
import mediapipe as mp

# POSITION DU CORPS : 

# Coté droit
RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
RIGHT_ELBOW = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
RIGHT_WRIST = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
RIGHT_HIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP
RIGHT_ANKLE = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
RIGHT_KNEE = mp.solutions.pose.PoseLandmark.RIGHT_KNEE
RIGHT_FOOT_INDEX = mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX

# Coté gauche
LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
LEFT_WRIST = mp.solutions.pose.PoseLandmark.LEFT_WRIST
LEFT_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP
LEFT_ANKLE = mp.solutions.pose.PoseLandmark.LEFT_ANKLE
LEFT_KNEE = mp.solutions.pose.PoseLandmark.LEFT_KNEE
LEFT_FOOT_INDEX = mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX


# Nom des exercices
EX_PUSH_UP = "Pompes"
EX_HEELS2BUTTOCKS = "Talons-fesses"
EX_KNEERAISE = "Montées de genoux"
EX_SQUATS = "Squats"
EX_PLANK = "Planche"

# MOUVEMENTS 
MOVE_UP = "move up"
MOVE_DOWN = "move down"
MOVE_UNKNWON = "unknown"
MOVE_LEFT_UP_RIGHT_DOWN = "left up right down"
MOVE_LEFT_DOWN_RIGHT_UP = "left down right up"
MOVE_LEFT_DOWN_RIGHT_DOWN = "left down right down"
MOVE_TREE_POSE = "tree pose"

# COULEURS BGR
BGR_NESSY_BLUE = (80, 48, 8)
BGR_NESSY_ORANGE = (74, 153, 242)

# Nom des fenêtres : ne pas mettre d'accent !
WIN_NAME_TITLE = "Bienvenue"
WIN_NAME_PUSHUP = "Pompes !"
WIN_NAME_HEELS2BUTTOCKS= "Talons fesses !"
WIN_NAME_KNEERAISE = "Montees de genoux !"
WIN_NAME_SQUATS = "Squats !"
WIN_NAME_PLANK = "Planche!"
WIN_NAME_TREE_POSE = "Tree pose !"

