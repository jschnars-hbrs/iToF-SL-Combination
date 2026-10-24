"""Start the GUI:  python -m gui   (from codePaperlike/).

Started with a Python that lacks the GUI packages (e.g. the system python3), it restarts itself with the
repository's .venv (Linux/macOS: .venv/bin/python, Windows: .venv\\Scripts\\python.exe).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
REQUIRED = ("PySide6", "pyvista", "pyvistaqt")

missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
if missing:
    venv = CODE_DIR.parent / ".venv"
    python = next((p for p in (venv / "bin" / "python", venv / "Scripts" / "python.exe") if p.exists()), None)
    if python is None or Path(sys.prefix).resolve() == venv.resolve():  # venv pythons are symlinks
        sys.exit(f"Missing packages {', '.join(missing)}. Install them with:\n"
                 f"  {sys.executable} -m pip install -r {CODE_DIR.parent / 'requirements.txt'}")
    print(f"({', '.join(missing)} not in {sys.executable} — starting with {python})")
    sys.exit(subprocess.call([str(python), "-m", "gui", *sys.argv[1:]], cwd=CODE_DIR))

sys.path.insert(0, str(CODE_DIR))

from gui.app import main  # noqa: E402

main()
