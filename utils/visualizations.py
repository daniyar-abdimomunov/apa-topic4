# %%
# Your first code cell (put imports or functions here)

# %%
# Another code cell (put more code here)


# This is a markdown cell. You can write notes here.*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any,Dict,Tuple
# %% [markdown]
# TO-DO: Create visualizations comparing metrics scores of different models
# %%
def _reduce_dimension(array: np.ndarray) -> np.ndarray:
    return array.mean(axis=1) if len(array.shape) > 1 else array
def compare_scores(benchmarking: Dict[str,Dict[str,Any]]) -> Tuple[plt.Figure, Any]:
    model_names=[]
    metrics_data=[]
    for model_key,model in benchmarking.items():
        name=model.get('name',model_key)
        metrics=model.get('metrics',{})
        model_names.append(name)
        metrics_data.append(metrics)
    df=pd.DataFrame(metrics_data,index=model_names)
    #one figure per metric
    figures={}
    for metric in df.columns:
        fig,ax=plt.subplots(figsize=(7,4))
        df[metric].plot(kind='bar',ax=ax)
        ax.set_title(f"{metric.upper()} Comparison Across Models")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())
        plt.xticks(rotation=0)
        plt.tight_layout()
        figures[metric]=(fig,ax)
    return figures



# TO-DO: Create visualizations comparing predictions of different models
def compare_single_prediction(y_input, y_true, benchmarking: dict, test_case: int) -> Tuple[plt.Figure, Any]:
    ...
    fig, ax = plt.subplots(figsize=(10,6))
    y_input = y_input[test_case]
    y_true = y_true[test_case]
    x_input = list(range(-y_input.shape[0], 0))
    x = list(range(y_true.shape[0]))

    plt.plot(x_input, y_input, 'k-',label='Input Data')
    plt.plot(x, y_true, '.', label='Observed Data', color='red')

    for model_key,model in benchmarking.items():
        is_probabilistic = 'lower' in model and 'upper' in model

        lower = _reduce_dimension(model['lower'][test_case]) if is_probabilistic else None
        upper = _reduce_dimension(model['upper'][test_case]) if is_probabilistic else None
        plt.fill_between(x=x, y1=lower, y2=upper, alpha=0.4) if is_probabilistic else None

        pred = _reduce_dimension(model['pred'][test_case])
        plt.plot(x, pred, label=model['name'])

    ax.set_title(f"Model Forecast comparison\n"
                 f"Test Case: #{test_case}")
    ax.set_ylabel("Price (EUR/MWhe)")
    ax.set_xlabel("Time-steps")
    plt.legend()
    plt.tight_layout()
    return fig, ax

def plot_predictions(
        input:np.ndarray,
        true:np.ndarray,
        pred:np.ndarray,
        lower:np.ndarray=None,
        upper:np.ndarray=None):
    x_input = list(range(-input.shape[0], 0))
    x_true = list(range(true.shape[0]))

    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Plot training data as black stars
    ax.plot(x_input, input, 'g')
    # Plot predictive means as blue line
    ax.plot(x_true, true, 'r.', alpha=0.5)
    # Plot predictive means as blue line
    ax.plot(x_true, pred, 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x=x_true, y1=lower, y2=upper, alpha=0.5)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price (EUR/MWhe)')
    #ax.set_xlim(-10, 20)
    ax.set_ylim([-20, 210])
    ax.legend(['Input Data', 'Observed Data', 'Prediction', 'Confidence Interval'])
    plt.show()
    return

def compare_multi_predictions(y_inputs, y_trues, benchmarking: dict, test_cases: list[int]) -> Tuple[plt.Figure, Any]:
    fig, axs = plt.subplots(nrows=len(benchmarking.keys()), ncols=len(test_cases),  figsize=(24,9), sharex=True, sharey=True, layout='tight')

    for i, (model_key, model) in enumerate(benchmarking.items()):
        for j, test_case in enumerate(test_cases):
            ax = axs[i, j]
            if j == 0:
                ax.set_ylabel(model['name'], size=18)
            if i == 0:
                ax.set_title(f'Test Case: #{test_case}', size=18)
            y_input = y_inputs[test_case]
            y_true = y_trues[test_case]
            x_input = list(range(-y_input.shape[0], 0))
            x = list(range(y_true.shape[0]))

            ax.plot(x_input, y_input, 'k-', label='Input Data')
            ax.plot(x, y_true, '.', label='Observed Data', color='red')

            if 'lower' in model and 'upper' in model:
                lower = _reduce_dimension(model['lower'][test_case])
                upper = _reduce_dimension(model['upper'][test_case])
                ax.fill_between(x=x, y1=lower, y2=upper, alpha=0.4, label='Confidence Interval')

            pred = _reduce_dimension(model['pred'][test_case])
            ax.plot(x, pred, label='Prediction')

    fig.suptitle(f"Model Forecast comparison", size=28)
    fig.supxlabel("Time-steps", size=18)
    plt.legend()
    plt.tight_layout()

    return fig, axs