# continuous-time-eco-models
 Deep neural network for solving continuous time economic models

## Start developing

### Code
All the code are undr [`ecomodels`](./ecomodels/), and the tests are under [`tests`](./tests/)

To run the code and tests locally

```
python -m venv venv
source venv/bin/activate # venv/Scripts/activate using Windows powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-doc.txt
```

For easier testing, you can create a file in the root folder of the project, and import functions from `ecomodels`.

To properly run all tests in the `tests/` folder
```
pip install -e .
pytest tests/
```

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
