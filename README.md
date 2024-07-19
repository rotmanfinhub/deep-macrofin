# Deep-MacroFin

Deep-MacroFin is a comprehensive deep-learning framework designed to solve partial differential equations, with a particular focus on models in continuous time economics. 

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
- [`macro_problems`](./examples/macro_problems/): 1D problem in the paper, with various parameters
- [`paper_example`](./examples/paper_example/): Other examples in the paper, with PyMacroFin and deepXDE comparisons
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
