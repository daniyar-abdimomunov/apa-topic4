import matplotlib.pyplot as plt
from typing import Any
from utils.metrics import _reduce_dimension

# TO-DO: Create visualizations comparing metrics scores of different models
def compare_scores(benchmarking: dict) -> (plt.Figure, Any):
    ...
    fig, ax = plt.subplots()
    return fig, ax


# TO-DO: Create visualizations comparing predictions of different models
def compare_predictions(x_input, y_input, x, y_true, benchmarking: dict) -> (plt.Figure, Any):
    ...
    fig, ax = plt.subplots()

    plt.plot(x_input, y_input, label='Input Data')
    plt.plot(x, y_true, 'r.', label='Observed Data')

    for model_name in benchmarking.keys():
        model = benchmarking[model_name]
        is_probabilistic = 'lower' in model.keys() and 'upper' in model.keys()

        lower = _reduce_dimension(model['lower']) if is_probabilistic else None
        upper = _reduce_dimension(model['upper']) if is_probabilistic else None
        plt.fill_between(x=x, y1=lower, y2=upper, alpha=0.4) if is_probabilistic else None

        pred = _reduce_dimension(model['pred'])
        plt.plot(x, pred, label=model['name'])

    ax.set_ylabel('Price (EUR/MWhe)')
    plt.legend()
    return fig, ax