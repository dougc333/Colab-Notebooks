import json
from pathlib import Path

REPO = "dougc333/Colab-Notebooks"
BRANCH = "main"

for nb_path in Path(".").rglob("*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    colab_url = (
        f"https://colab.research.google.com/github/"
        f"{REPO}/blob/{BRANCH}/{nb_path.as_posix()}"
    )

    badge_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "[![Open In Colab]"
            "(https://colab.research.google.com/assets/colab-badge.svg)]"
            f"({colab_url})\n"
        ],
    }

    # Avoid duplicates
    if nb["cells"] and badge_cell["source"][0] in "".join(nb["cells"][0].get("source", [])):
        continue

    nb["cells"].insert(0, badge_cell)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Updated {nb_path}")
