from typing import Union, Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def obtain_indexes_from_category_item(ser_cate : pd.Series,
                                      ser_item : pd.Series
                                      ) -> Tuple[np.array, np.array]:
    """Create index for category and item from series of 
    category and item for vertically aligned labels and errorbar plot.
    It is noted that index has negative continuous values, 
    starting from 0 to -n.

    Args:
        ser_cate: Series of categories matched with ser_item.
        ser_item: Series of items.

    Returns:
        Indexes for caategories and items.
    """
    ser_cate = ser_cate.copy()

    n_cate = len(ser_cate.unique())
    ser_cate[ser_cate.duplicated()] = np.nan
    y_index = []
    y_index_cate = []
    index = 0
    for cate, item in zip(ser_cate, ser_item):
        if cate != cate:
            y_index.append(index)
        else:
            y_index_cate.append(index)
            index -= 1
            y_index.append(index)
        index -= 1

    y_index_cate = np.array(y_index_cate)
    y_index = np.array(y_index)

    return(y_index_cate, y_index)


def errorbar_forestplot(
    ax: plt.Axes,
    y_index: np.array,
    df: Optional[pd.DataFrame] = None,
    risk: str = "risk",
    lower: Union[str, int] = 0,
    upper: Union[str, int] = 1,
    y_adj: float = 0,
    errorbar_kwds: Optional[dict] = None,
    ref_kwds: Optional[dict] = None,
    errorbar_color: Optional[str] = None,
    ref_color: Optional[str] = None,
    label: Optional[str] = None,
    log_scale: bool = False,
    ):
    """Error bar plot for a forest plot.

    Args:
        ax: Axis to be drawn.
        y_index: index to be plotted.
        risk: Column name for risk.
        lower: Column name for lower confidence interval.
        upper: Column name for upper confidence interval.
        y_adj: For this value, plotting is moved.
        errorbar_kwds: Passed to ax.errorbar function.
        ref_kwds: Passed to ax.scatter function.
        df: Dataframe for another result.
        label: Label for stratified drawings. Passed to ax.errorbar.
        log_scale: Plot risk in log scale (np.log).
    """
    y_index = y_index + y_adj

    df = df.copy()
    if errorbar_kwds is None:
        errorbar_kwds = dict(fmt="o",
                             capsize=5,
                             markeredgecolor="black",
                             ecolor="black",
                             color='white'
                             )
    if ref_kwds is None:
        ref_kwds = dict(marker="s", s=20, color="black")

    if log_scale:
        df[risk] = np.log(df[risk])
        df[lower] = np.log(df[lower])
        df[upper] = np.log(df[upper])

    df["xerr_lower"] = df[risk] - df[lower]
    df["xerr_upper"] = df[upper] - df[risk]

    cond = df[risk].notnull()
    ax.errorbar(df.loc[cond, risk],
                y_index[cond],
                xerr=df.loc[cond, ["xerr_lower", "xerr_upper"]].T,
                label=label,
                zorder=5,
                **errorbar_kwds
                )

    cond = df[risk].isnull()
    ref_v = 0 if log_scale else 1
    df["ref"] = df[risk].mask(cond, ref_v).mask(~cond, np.nan)
    ax.scatter(df["ref"], y_index, zorder=5, **ref_kwds)


def embed_strings_forestplot(
    ax: plt.Axes,
    ser: pd.Series,
    y_index: np.array,
    x: float,
    header: str,
    fontsize: int = None,
    y_header: float = 1.0,
    text_kwds: Optional[dict] = None,
    header_kwds: Optional[dict] = None,
    replace: Optional[dict] = None,
    ):
    """Embed strings/values of one column with header.

    Args:
        ser: Values of this series will be embedded.
        x: x axis value of text position, ranging from 0 to 1.
        df: Dataframe for another result.
    """
    if text_kwds is None:
        text_kwds = {}
    if header_kwds is None:
        header_kwds = {}
    ax.text(x, y_header, header, ha="left", va="center",
            fontsize=fontsize, **header_kwds)

    if replace is not None:
        ser = ser.replace(replace)

    for y, text in zip(y_index, ser):
        ax.text(x, y, text, ha="left", va="center",
                fontsize=fontsize, **text_kwds)

