# Ca serait bien de mettre ici toutes les constantes du projets,
# par exemple les positions des points qui sont utiles
import mediapipe as mp

# Résolution de l'écran utilisée pour le devellopement
SCREEN_H_DEV = 720
SCREEN_W_DEV = 1280

# Résolution de l'écran pour l'utilisateur
SCREEN_H = 720
SCREEN_W = 1280

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
EX_HEELS2BUTTOCKS = "Talons-fesses"
EX_KNEERAISE = "Montées de genoux"
EX_SQUATS = "Squats"

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
BGR_NESSY_LIGHT_GREY = (239, 239, 239)

# Nom des fenêtres : ne pas mettre d'accent !
WIN_NAME_TITLE = "Bienvenue"
WIN_NAME_PUSHUP = "Pompes !"
WIN_NAME_HEELS2BUTTOCKS= "Talons fesses !"
WIN_NAME_KNEERAISE = "Montees de genoux !"
WIN_NAME_SQUATS = "Squats !"
WIN_NAME_TREE_POSE = "Tree pose !"