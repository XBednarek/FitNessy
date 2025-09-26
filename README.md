# FitNessy
Application de fitness

# Pour générer l'env. python :
uv sync

# Et pour lancer l'application :
uv run python3 -m fitness.app

# Si veut run juste un module du package fitnessy : (exemple avec app)
uv run python3 -m fitness.app

# Et si on veut en plus se passer de tous les warnings ...
uv run python3 -m fitness.app 2>/dev/null

# On peut aussi activer l'env. (un peu comme on faisait avec conda activate) :
source .venv/bin/activate