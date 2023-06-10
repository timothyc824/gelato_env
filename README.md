# Gelateria

...

## Installation

### Linux

It is convenient to make use of `pipx` to install general helper packages:

```bash
python -m venv $HOME/.venvs
source $HOME/.venvs/bin/activate
pip install pipx
pipx install black
pipx install isort
pipx install pydocstyle
pipx install ruff
pipx install pre-commit
```

Use the Makefile to install the repo and its dependencies:

```bash
make setup
```
