# Deep-MacroFin

[![Build Status](https://github.com/rotmanfinhub/deep-macrofin/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/rotmanfinhub/deep-macrofin/actions/workflows/ci-build.yaml)
[![PyPI version](https://badge.fury.io/py/deep-macrofin.svg)](https://badge.fury.io/py/deep-macrofin)
[![PyPI Downloads](https://static.pepy.tech/badge/deep-macrofin)](https://pepy.tech/project/deep-macrofin)
[![PyPI - License](https://img.shields.io/pypi/l/deep-macrofin)](https://github.com/rotmanfinhub/deep-macrofin/blob/main/LICENSE)

Deep-MacroFin is a comprehensive deep-learning framework designed to solve equilibrium economic models in continuous time. The library leverages deep learning to alleviate curse of dimensionality.

**Documentation:** [mkdocs](https://rotmanfinhub.github.io/deep-macrofin)

## Start developing

### Code
All the code are under [`deep_macrofin`](./deep_macrofin/), and the tests are under [`tests`](./tests/)

To run the code and tests locally

```
python -m venv venv
source venv/bin/activate # venv/Scripts/activate using Windows powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-doc.txt
```

For easier testing, you can create a file in the root folder of the project, and import functions from `deep_macrofin`.

To properly run all tests in the `tests/` folder
```
pip install -e .
pytest tests/
```

### Examples
Various examples using the library, with comparisons to DeepXDE and PyMacroFin are included in [`examples`](./examples/)

- [`basic_examples`](./examples/basic_examples/): Solutions to basic ODEs/PDEs, diffusion equation, function approximation and systems of ODEs, with some comparisons to DeepXDE.
- [`initial_examples`](./examples/initial_examples/): Initial scripts for testing deep neural networks for ODE/PDE solutions, and macromodels.
- [`kan_examples`](./examples/kan_examples/): Solutions to basic ODEs, using KAN as approximators
- [`macro_problems`](./examples/macro_problems/): Macroeconomic models in different dimensions.
- [`paper_example`](./examples/paper_example/): Examples in the paper, with PyMacroFin and deepXDE comparisons. Models and log files for reproducing paper results can be found in [Google Drive](https://drive.google.com/drive/folders/1wVtO9JUq_a7IhA9Sult2oYmKOX5GHcPh?usp=sharing).
- [`pymacrofin_eg`](./examples/pymacrofin_eg/): Examples from PyMacroFin and proposition 2 from Brunnermeier and Sannikov (2014)

**Note**:  <a href="https://adriendavernas.com/pymacrofin/index.html" target="_blank">PyMacroFin</a> and <a href="https://github.com/lululxvi/deepxde/tree/master" target="_blank">DeepXDE</a> are used as benchmarks in several examples, but the associated packages are not included in this repo's `requirements.txt`. To run the comparisons properly, please install their packages respectively.

### Docs
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

### Cite Deep-MacroFin

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
