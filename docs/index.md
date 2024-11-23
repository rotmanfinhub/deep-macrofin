# Deep-MacroFin

Deep-MacroFin is a comprehensive deep-learning framework designed to solve equilibrium economic models in continuous time. The library leverages deep learning to alleviate curse of dimensionality.

## Installation

### Install from PyPI

The stable version of the package can be installed from PyPI.

```bash
pip install deep-macrofin
```

### Build from Source (with poetry)
The project is now configured with [poetry](https://python-poetry.org/) for dependency management and packaging. 
To install the dependencies and run the code:

1. Clone the repository
```bash
git clone https://github.com/rotmanfinhub/deep-macrofin.git
```

2. Install poetry by following the official documentation [here](https://python-poetry.org/docs/#installation)

3. Create a poetry virtual environment and install the dependencies and the package
```bash
poetry config virtualenvs.in-project true --local # this sets the virtual environment path to be in the local directory.
poetry shell # creates the virtual environment
poetry install --no-interaction # installs the dependencies and the package
```

### Build from Source (without poetry)

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

## Cite Deep-MacroFin

If you use Deep-MacroFin for academic research, you are encouraged to cite the following paper:

```
@misc{wu2024deepmacrofin,
      title={Deep-MacroFin: Informed Equilibrium Neural Network for Continuous Time Economic Models}, 
      author={Yuntao Wu and Jiayuan Guo and Goutham Gopalakrishna and Zisis Poulos},
      year={2024},
      eprint={2408.10368},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.10368}, 
}
```

<!-- [^1]: Adrien d'Avernas and Damon Petersen and Quentin Vandeweyer, *"Macro-financial Modeling in Python: PyMacroFin"*, 2021-11-18  
[^2]: Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em, *"DeepXDE: A deep learning library for solving differential equations"*, SIAM Review, 63(1): 208–228, 2021 -->