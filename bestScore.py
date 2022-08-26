import pathlib
import os

path = '/home/brian.atkinson/Bayes/data/PenaltyTestSearch'
dataDir = pathlib.Path(path) if not isinstance(path,pathlib.PosixPath) else path

best_episode = -1
best_score = -1
best_dir = None

for dir in dataDir.iterdir():
    try:
        with open(os.path.join(dir,'digest.txt'),'r') as f:
            lines = f.readlines()
        score = int(lines[1].split(':')[1].lstrip(' '))
        if score > best_score:
            episode = int(lines[0].split(':')[1].lstrip(' '))
            if episode > best_episode:
                best_score = score
                best_dir = dir
                best_episode = episode
    except FileNotFoundError:
        continue

print(f'best dir: {best_dir}')
print(f'best score: {best_score}')
print(f'best episode: {best_episode}')