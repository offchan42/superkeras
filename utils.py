import numpy as np
import pandas as pd
from keras.utils import to_categorical
import json


def load_multi_csv(filenames, concat=True, discard_empty=False, verbose=True):
    """
    Load CSV data from multiple files and concatenate as DataFrame

    Args:
        filenames: A list of strings or file-like objects
        concat: If False, return a list instead of DataFrame.
        discard_empty: If True, all empty DataFrames will be discarded.
        verbose: Whether to print loaded file stats or not

    Returns:
        A list or a DataFrame
    """
    df_list = []
    for fn in filenames:
        df = pd.read_csv(fn)
        print_args = [fn, df.shape]
        if discard_empty and df.empty:
            print_args.append("- discarded")
        else:
            df_list.append(df)
        if verbose:
            print(*print_args)
    if not concat:
        #         print([df.shape for df in df_list])
        return df_list
    thedf = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    if verbose:
        print(thedf.shape)
    return thedf


def make_xy_3d(
    big_df,
    numsteps,
    skip_size=1,
    categorical=True,
    add_time_y=False,
    y_dim=1,
    num_classes=None,
):
    """
    Extract `X` and y` chunks from `big_df` assuming `big_df` is a time-series
    data.
    
    Convert DataFrame of shape (?, n_features) into 3D array of shape
    (n_samples, numsteps, n_features-y_dim) for X.

    If y_dim=1, shape of y will be (n_samples,). If `categorical` is also True
    then the shape will be (n_samples, n_classes).

    If y_dim>1, shape of y will be (n_samples, y_dim).

    If y_dim=0, return None in the place of y.

    If `add_time_y` is True, an extra time dimension will be added.

    `num_classes` should be provided when the data is too few that the amount
    of classes cannot be inferred for `to_categorical` function."""
    X, y = [], []
    if isinstance(big_df, pd.DataFrame):
        big_df = big_df.values
    for i in range(0, big_df.shape[0], skip_size):
        arr = big_df[i : i + numsteps]  # 2d array of shape (numsteps, n_features)
        if arr.shape[0] != numsteps:
            break
        if y_dim == 0:
            X.append(arr[np.newaxis, :, :])
        else:
            X.append(arr[np.newaxis, :, :-y_dim])
            if y_dim == 1:
                y.append(arr[:, -1] if add_time_y else arr[-1, -1])
            else:
                y.append(arr[:, -y_dim:] if add_time_y else arr[-1, -y_dim:])
    if y_dim == 0:
        return np.concatenate(X), None
    return (
        np.concatenate(X),
        to_categorical(np.array(y), num_classes=num_classes)
        if categorical
        else np.array(y),
    )


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def make_json_serializable(obj, serializer=None, inplace=False):
    """
    Ensure the object will be serializable by converting the non-serializable part to be serializable.
    If `inplace` then will mutate the object directly.
    `serializer` is the function that receives the non-serializable object and return a serializable object.
    If it is None, the default `serializer` will be something that converts the object to string
    """
    if isinstance(obj, dict):
        serialized_obj = obj if inplace else dict()
        for key, val in obj.items():
            serialized_obj[key] = make_json_serializable(
                val, serializer=serializer, inplace=inplace
            )
        return serialized_obj
    elif isinstance(obj, list):
        serialized_obj = obj if inplace else list()
        for i, val in enumerate(obj):
            val = make_json_serializable(val, serializer=serializer, inplace=inplace)
            if inplace:
                serialized_obj[i] = val
            else:
                serialized_obj.append(val)
        return serialized_obj
    elif is_jsonable(obj):
        return obj
    else:
        if serializer is None:

            def serializer(x):
                if hasattr(x, "name"):
                    return x.name
                if hasattr(x, "__name__"):
                    return x.__name__
                return type(x).__name__

        obj_ser = serializer(obj)
        print("serialize", repr(obj), "as", repr(obj_ser))
        return obj_ser


def summarize_to_list(model, **summary_kwargs):
    """
    Call `model.summary()` and append each line to a list.
    Then return the list.
    """
    summary = []
    add_to_summary = lambda line: summary.append(line)
    model.summary(print_fn=add_to_summary, **summary_kwargs)
    return summary
