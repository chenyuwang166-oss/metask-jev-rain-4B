"""Release members, pruning clone metadata and generated local directories."""
import os
from pathlib import Path

IGNORED={'.git','__pycache__','state'}

def package_files(root):
    root=Path(root)
    for directory, dirs, names in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in IGNORED)
        for name in dirs:
            if (Path(directory)/name).is_symlink():
                raise ValueError('Symlink directory in package')
        for name in sorted(names):
            if name not in IGNORED:
                yield Path(directory)/name
