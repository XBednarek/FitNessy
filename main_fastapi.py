# Pour lancer ce fichier :
# uv run uvicorn main_fastapi:app --host 127.0.0.1 --port 8789
# Puis ouvrir dans un navigateur : http://localhost:8789/

"""Script pour lancer l'application de fitess via fast API"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from threading import Thread
import queue
from fitness.app import App
import fitness.constants as cst

# Application Fast API
app = FastAPI()

# Queue pour passer les frames entre threads
# Pour les images
frame_queue = queue.Queue(maxsize=2)
# Pour les actions de l'utilisateurs
action_queue = queue.Queue()

# Fonction qui démarre l'application de fitness
def startfitness():
    # Création de l'app :
    fitnessapp = App(verbose=True, fast_api_queues=(frame_queue, action_queue))
    # Lancement de l'app :
    exos = {cst.EX_PLANK: 15,cst.EX_SQUATS: 3, cst.EX_PUSH_UP: 3,cst.EX_PLANK: 5}
    fitnessapp.run_exercice_session(exos)

# Générateur de frame : lecture du buffer (lui même remplit par l'appli de fitness)
def generate_frames():
    """Générateur qui yield les frames au format MJPEG"""
    while True:
        try:
            # Récupérer une frame de la queue
            frame_bytes = frame_queue.get(timeout=1)
            
            # Format MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except queue.Empty:
            # Pas de nouvelle frame disponible
            continue

# Que faire quand on lance fast api ?
@app.on_event("startup")
async def startup_event():
    """Démarrer la capture vidéo au lancement"""
    capture_thread = Thread(target=startfitness, daemon=True)
    capture_thread.start()

# Lecture du flux vidéo dans le navigateur :
@app.get("/video_feed")
async def video_feed():
    """Endpoint pour le flux vidéo"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/action")
async def send_action(request: Request):
    """Reçoit l'action du bouton"""
    data = await request.json()
    action_queue.put(data["message"])
    return {"status": "ok"}

# HTML d'affichage
@app.get("/", response_class=HTMLResponse)
async def index():
    """Page HTML pour afficher le flux"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flux Vidéo FastAPI</title>
    </head>
    <body>
        <h1>Flux Vidéo en Direct</h1>
        <img src="/video_feed" width="640" height="480">
        <br>
        <button onclick="sendAction()">Next</button>

        <script>
            async function sendAction() {
                await fetch('/action', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: "quit"})
                });
            }
        </script>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
