# Aurora Evolutions

Aurora Evolutions is an interactive 2D evolution sandbox inspired by Conway's Game of Life. Instead of simple binary states, organisms carry a small "genetic matrix" that drives their behaviour, environmental tolerance, diet and appearance. The world also simulates different terrain materials, temperature and light cycles, creating a small ecosystem with food chains and emergent species.

## Features

- **Dynamic environment** – soil, water, rock and sand tiles each influence light levels, fertility and temperature. A day/night cycle shifts global light and temperature across the map.
- **Living vegetation layer** – fertile tiles grow vegetation that herbivores consume while phototrophs feed directly from available light.
- **Genetic matrices** – each lifeform stores a 3×3 matrix of values. These genes are interpreted into traits such as diet, preferred biome, metabolism, reproduction cost and mutation chance.
- **Food chains** – phototrophs collect light, herbivores graze on vegetation and carnivores hunt other creatures. Omnivorous species can mix strategies.
- **Mutation and speciation** – when organisms reproduce their matrices mutate, automatically creating new species with generated names and colours when the genes diverge enough.
- **In-world encyclopedia** – press `E` at any time to inspect discovered species, their traits, population history and genetic matrices.
- **Creative sandbox tools** – paint different materials, seed new organisms and remix the world while the simulation runs.

## Controls

| Action | Input |
| --- | --- |
| Pause / resume simulation | `Space`
| Toggle species encyclopedia overlay | `E`
| Reset world | `R`
| Paint terrain material | `1` Soil, `2` Water, `3` Rock, `4` Sand (hold the number and left-click)
| Add lifeforms | `P` Phototroph, `H` Herbivore, `C` Carnivore, `O` Omnivore (then left-click a tile)
| Scatter random organisms | `L`
| Pan encyclopedia list while open | Arrow keys or mouse wheel |
| Quit | `Esc` or window close button |

## Getting started

1. Install Python 3.10 or newer.
2. (Optional but recommended) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the simulator:
   ```bash
   python -m src.main
   ```

## Building a standalone executable (Windows)

You can bundle the project into a single executable using [PyInstaller](https://pyinstaller.org/):

```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --name "Aurora Evolutions" --add-data "assets:assets" src/main.py
```

This command creates `dist/Aurora Evolutions.exe`. Copy the `assets` folder (if you add custom art or data) alongside the executable. On other platforms you can use the same PyInstaller command or the platform-specific packagers you prefer.

## Project structure

```
.
├── requirements.txt
├── readme.md
└── src
    └── main.py
```

Feel free to fork the repository, tweak genetic interpretation rules, add new materials or extend the UI. The code is heavily commented to help you navigate and expand the simulation.
