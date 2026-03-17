import re


def save_results_with_params(
    df, backup, params, folder="results"
):
    """
    Save a dataframe to CSV with a filename that encodes the parameters.
    """
    import os
    os.makedirs(folder, exist_ok=True)

    # Build filename string
    param_parts = []
    for k, v in params.items():
        # Format floats with 3 significant digits, ints as is
        if isinstance(v, float):
            v_str = f"{v:.3g}"
        else:
            v_str = str(v)
        # Replace unsafe filename characters
        v_str = re.sub(r"[^\w\d]+", "_", v_str)
        param_parts.append(f"{k}_{v_str}")

    filename = f"{backup}_" + "_".join(param_parts) + ".csv"
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
