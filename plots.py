def plot_predictions(y_actual, y_pred, dataset_name="validation", figsize=(12, 6)):
    """
    Plot predicted values against actual values.
    
    Parameters:
    -----------
    y_actual : array-like
        Actual target values
    y_pred : array-like
        Predicted values to compare against actual values
    dataset_name : str, default="validation"
        Name of the dataset being plotted (e.g., "validation" or "test")
    figsize : tuple, default=(12, 6)
        Figure size as (width, height)
    
    Returns:
    --------
    None : Displays a plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create a time index for x-axis (if not datetime, just use sequential numbers)
    if hasattr(y_actual, 'index') and hasattr(y_actual.index, '__class__') and 'DatetimeIndex' in y_actual.index.__class__.__name__:
        time_index = y_actual.index
    else:
        time_index = np.arange(len(y_actual))
    
    # Plot actual values
    plt.plot(time_index, y_actual, label='Actual', color='blue', linewidth=2)
    
    # Plot predicted values
    plt.plot(time_index, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
    
    # Add labels and title
    plt.title(f'Predictions vs Actual Values on {dataset_name.capitalize()} Set')
    plt.xlabel('Time' if hasattr(time_index, '__class__') and 'DatetimeIndex' in time_index.__class__.__name__ else 'Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0 for reference if data crosses zero
    if min(y_actual) < 0 < max(y_actual) or min(y_pred) < 0 < max(y_pred):
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Show the plot
    plt.tight_layout()
    plt.show()