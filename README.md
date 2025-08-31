# Seminar Applied Predictive Analytics: Topic 4
## Using Gaussian Processes for Probabilistic, Non-Parametric Electricity Price Forecasting

Authors: Daniyar Abdimomunov, Nourcéne Brahem, Oscar Eduardo Jaume Castro, Vasily Schob 
GitHub Repository: https://github.com/daniyar-abdimomunov/apa-topic4

This repository explores possible extensions of Gaussian Processes to make multi-step probabilistic predictions for time-series data, specifically working with EU27 hourly electricity prices.
Also, we compare the extended GP models to other time-series models for benchmarking.

This project is based on Python version 3.12. Please run ```pip install -r requirements.txt``` before working further with the repository.

This repository is organized in the following structure:
- the 'data' directory includes the raw electricity price data, as well as the processed data and predictions from trained models.
- the 'utils' directory includes all the functions and classes used by the Training and Benchmarking notebooks.
  - the 'models' sub-directory includes the classes of the models trained as part of this project.
- the 'scripts' directory includes all the notebooks store as Python scripts.
  - in order to convert an individual .py Python script file (e.g. Training.py) into a .ipynb Jupyter Notebook run the following command in Terminal in the project root directory: ```jupytext --sync scripts/Training.py```
  - in order to convert all .py Python script files to .ipynb Jupyter Notebook files, you can run the following command in Terminal in the project root directory (result may be inconsistent on non-UNIX systems): ```jupytext --sync scripts/*.py```
  - for more details, see the README.md file in the 'scripts' directory.

