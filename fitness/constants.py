# Ca serait bien de mettre ici toutes les constantes du projets,
# par exemple les positions des points qui sont utiles
import mediapipe as mp

# Résolution de l'écran utilisée pour le devellopement
SCREEN_H_DEV = 720
SCREEN_W_DEV = 1280

# Résolution de l'écran pour l'utilisateur
SCREEN_H = 360
SCREEN_W = 640

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
EX_KNEERAISE = "Montees de genoux"
EX_SQUATS = "Squats"
EX_TREEPOSE = "Tree Pose"
EX_MEDITATION_POSE = "Meditation pose"
EX_PLANK = "Planche"
EX_COBRA="cobra"
# MOUVEMENTS 
MOVE_UP = "move up"
MOVE_DOWN = "move down"
MOVE_UNKNWON = "unknown"
MOVE_LEFT_UP_RIGHT_DOWN = "left up right down"
MOVE_LEFT_DOWN_RIGHT_UP = "left down right up"
MOVE_LEFT_DOWN_RIGHT_DOWN = "left down right down"
MOVE_TREE_POSE = "tree pose"
MOVE_MEDITATION_POSE = "meditation pose"

# COULEURS BGR
BGR_NESSY_BLUE = (80, 48, 8)
BGR_NESSY_ORANGE = (74, 153, 242)
BGR_NESSY_LIGHT_GREY = (239, 239, 239)
BGR_RED_FILTER = (255,0,0)

# COULEURS RGB
RGB_NESSY_BLUE = (8, 48, 80)
RGB_NESSY_ORANGE = (242, 153, 74)
RGB_NESSY_LIGHT_GREY = (239, 239, 239)
RGB_DIFFICULTY_MID_GREY = (140, 139, 139)
RGB_BLACK_TEXT = (0,0,0)


# Nom des fenêtres : ne pas mettre d'accent !
WIN_NAME_DEFAULT = "MediaPipe"
WIN_NAME_TITLE = "Bienvenue"
WIN_NAME_PUSHUP = "Pompes !"
WIN_NAME_HEELS2BUTTOCKS= "Talons fesses !"
WIN_NAME_KNEERAISE = "Montees de genoux !"
WIN_NAME_SQUATS = "Squats !"
WIN_NAME_MEDITATION_POSE = "Meditation pose !"
WIN_NAME_PLANK = "Planche!"
WIN_NAME_COBRA = "Cobra!"
WIN_NAME_TREE_POSE = "Tree pose !"

# Durée de la fenètre de félicitations
CONGRATS_DURATION_S = 5 # En secondes 
