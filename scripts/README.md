# Version Control of Jupyter Notebooks
Version controlling for Jupyter notebooks (.ipynb) is done using the **Jupytext** package. 
When using JupyterLab, Jupytext automatically syncs changes made to any .ipynb file in the `notebooks/` directory to a paired .py file in the `scripts/` directory.

When cloning the repository for the first time the `notebooks/` directory will not exist and the `scripts/` directory will need to be synced manually using a terminal command (see below). 
This will create a local `notebooks/` directory with notebooks paired to the .py files.

When pulling changes, Jupytext will sync pulled .py files in the `scripts/` directory and automatically update the paired .ipynb files in the local `notebooks/` directory.
The notebook needs to be reloaded for changes to take place.

When commiting  and pushing changes to Git, only commit changes made to the .py files in the `scripts/` directory. 
The `notebooks/` directory is automatically ignored using the `.gitignore` file.

## Manual syncing with Jupytext
If Jupytext doesn't sync automatically or paired files do not exist locally, run this command in terminal to manually convert .py files to .ipynb files:

```jupytext --sync scripts/*.py```

To convert an individual .py file (e.g. Training.py) run:

```jupytext --sync scripts/Training.py```

To convert .ipynb files to .py files run:

```jupytext --sync notebooks/*.ipynb```

To convert an individual .ipynb files (e.g. Training.ipynb) run:

```jupytext --sync notebooks/Training.ipynb```
