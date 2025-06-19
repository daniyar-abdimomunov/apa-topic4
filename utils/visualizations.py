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
def compare_predictions(x_input, y_input, x, y_true, benchmarking: dict) -> Tuple[plt.Figure, Any]:
    ...
    fig, ax = plt.subplots(figsize=(10,6))

    plt.plot(x_input, y_input, label='Input Data',linestyle='--',color='gray')
    plt.plot(x, y_true, 'k-', label='Observed Data')

    for model_key,model in benchmarking.items():
        pred = _reduce_dimension(model['pred'] )
        is_probabilistic = 'lower' in model and 'upper' in model

        lower = _reduce_dimension(model['lower']) if is_probabilistic else None
        upper = _reduce_dimension(model['upper']) if is_probabilistic else None
        plt.fill_between(x=x, y1=lower, y2=upper, alpha=0.4) if is_probabilistic else None

        pred = _reduce_dimension(model['pred'])
        plt.plot(x, pred, label=model['name'])
    ax.set_title("Model Forecast comparison")
    ax.set_ylabel('Price (EUR/MWhe)')
    ax.set_xlabel("time")
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