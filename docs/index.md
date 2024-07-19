# Deep-MacroFin

Deep-MacroFin is a comprehensive deep-learning framework designed to solve partial differential equations, with a particular focus on models in continuous time economics. 
It is inspired from <a href="https://adriendavernas.com/pymacrofin/index.html" target="_blank">PyMacroFin</a>[^1] and <a href="https://github.com/lululxvi/deepxde/tree/master" target="_blank">DeepXDE</a>[^2] 

## Installation

### Install from PyPI

The stable version of the package can be installed from PyPI.

```bash
pip install deep-macrofin
```

### Build from Source

For developers, you should clone the folder to your local machine and install from the local folder.

1. Clone the repository
```bash
git clone https://github.com/rotmanfinhub/deep-macrofin.git
```

2. Create a virtual environment (Optional, but recommended)
```bash
python -m venv venv
source venv/bin/activate # venv/Scripts/activate using Windows powershell
```

3. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-doc.txt
```

4. Install the package
```bash
pip install -e .
```


#### Docs
The documentation site is based on [mkdocs](https://www.mkdocs.org/) and [mkdocs-mateiral](https://squidfunk.github.io/mkdocs-material/).

Layouts
```
mkdocs.yml    # The configuration file.
docs/
    index.md  # The documentation homepage.
    ...       # Other markdown pages, images and other files.
```

To see the site locally, run the following command:
```
mkdocs serve
```


[^1]: Adrien d'Avernas and Damon Petersen and Quentin Vandeweyer, *"Macro-financial Modeling in Python: PyMacroFin"*, 2021-11-18  
[^2]: Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em, *"DeepXDE: A deep learning library for solving differential equations"*, SIAM Review, 63(1): 208–228, 2021