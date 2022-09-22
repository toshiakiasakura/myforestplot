from typing import Union, Optional, List, Dict, Tuple, Any, Callable
import pandas as pd
import numpy as np
import statsmodels


def statsmodels_fitting_result_dataframe(
    res,
    alpha: float = 0.05,
    accessor: Callable[[np.array], np.array] = np.exp,
    ) -> pd.DataFrame:
    """Create category and item columns from the statsmodels result.
    Categorical results are divided into original column name (category) and
    its items (item).

    Args:
        res: statsmodels' fitting results.
        alpha: The significance level for the confidence interval.
        accessor: Function to access each model result, which is summarized and displayed.
    """
    df = accessor(res.conf_int(alpha=0.05))
    df["risk"] = accessor(res.params)
    df["pvalues"] = res.pvalues

    cate = "category"
    df[cate] = np.nan
    rename_dic = {}
    for ind in df.index:
        if "[" in ind:
            s1, s2 = ind.split("[")
            rename_dic[ind] = s2[2:-1]
            # For case of specifying Treatment in formula.
            if "Treatment('" in s1:
                s1 = s1.split(",")[0][2:]
            df.loc[ind, cate] = s1
    df.rename(index=rename_dic, inplace=True)

    # Insert the same name for "category" in case of continuous variables.
    cond = df[cate].isnull()
    df[cate] = df[cate].mask(cond, df.index)
    df = df.reset_index().rename(columns={"index":"item"})

    df.insert(0, column="category", value=df.pop("category"))
    df.insert(1, column="item", value=df.pop("item"))

    # drop Intercept.
    df = df[df["category"] != "Intercept"]

    return df


def add_pretty_risk_column(res: pd.DataFrame, risk: str, lower: str, upper: str,
                           fml: str = ".2f", ref: str = "Ref."
                           ) -> pd.Series:
    """Add prrety risk string column.

    Args:
        res: Dataframe contaning points and confidence intervals.
        risk: point estimates of risk column name.
        lower: lower confidence interval column name.
        upper: upper confidence interval column name.
        fml: formula for f string.
        ref: if point esitmate column is empty, insert this string.
    """
    def f(x):
        risk_v = x[risk]
        lower_v = x[lower]
        upper_v = x[upper]
        s = f"{risk_v:{fml}} ({lower_v:{fml}}, {upper_v:{fml}})"
        return s

    ser = (res.apply(f, axis=1)
           .mask(res[risk].isnull(), ref)
           )
    return ser


def count_category_frequency(df: pd.DataFrame,
                             categorical_cols: List[str],
                             impute_continuous: bool = True,
                             ) -> pd.DataFrame:
    """Count category frequency.

    Args:
        df: Original dataframe.
        categorical_cols: Columns for categorical variables.
        impute_continuous: columns not specified as categorical_cols were
            imputed for item and number of observations (nobs).
    """
    n = df.shape[0]
    sers = [(df[c]
             .value_counts()
             .to_frame()
             .stack()
             )
             for c in categorical_cols]
    ser_sum = pd.concat(sers)
    df_nobs = (ser_sum
               .reset_index()
               .rename(columns={"level_0": "item",
                                "level_1": "category",
                                0:"nobs"}
                       )
               )

    df_nobs.insert(0, column="category", value=df_nobs.pop("category"))
    df_nobs.insert(1, column="item", value=df_nobs.pop("item"))
    return df_nobs


def sort_category_item(df_: pd.DataFrame,
                       order: List[str],
                       item_order: Dict[str, List[str]] = None,
                       ) -> pd.DataFrame:
    """Sort category and item based on categorical values.

    Args:
        df_ : dataframe containing category and item.
        categorical: Dictionary containing column names and its order of items.
        order : if specified, category is ordered based on this variable.

    """
    if item_order is None:
        item_order = {}

    df_sorted = pd.DataFrame()
    for c in order:
        cond = df_["category"] == c
        dfM = df_[cond]
        if c in item_order.keys():
            lis = item_order[c]
            sort_dic = {l:i for i,l in enumerate(lis)}
            dfM = dfM.sort_values(by="item",key=lambda x: x.replace(sort_dic))
        df_sorted = pd.concat((df_sorted, dfM), axis=0)
    return df_sorted


def statsmodels_pretty_result_dataframe(
    data: pd.DataFrame,
    res,
    order: List[str],
    cont_cols: Optional[List[str]] = None,
    item_order: Dict[str, List[str]] = None,
    fml: str = ".2f",
    accessor: Callable[[np.array], np.array] = np.exp,
) -> pd.DataFrame:
    """Obtain pretty result dataframe from statsmodels results.
    Fitting coefficients are converted by np.exp.

    Args:
        data: original dataframe.
        res: statsmodels results.
        categorical: Dictionary containing column names and its order of items.
        order : if specified, category is ordered based on this variable.
        fml: formula for f string of pretty risk.
        accessor: Function to access each model result, which is summarized and displayed.
    """
    if res.nobs != data.shape[0]:
        raise Exception(("Some observations were dropped when fitted, "
                         "check number of observations"
                        ))
    if cont_cols is None:
        cate_cols = order
    else:
        cate_cols = [c for c in order if not c in cont_cols]
    df_res = statsmodels_fitting_result_dataframe(res, alpha=0.05, accessor=accessor)
    df_nobs = count_category_frequency(data, cate_cols)
    df_sum = pd.merge(df_res, df_nobs,
                      on=["category", "item"],
                      validate="1:1",
                      how="outer")
    df_sum = sort_category_item(df_sum, order=order, item_order=item_order)
    df_sum["risk_pretty"] = add_pretty_risk_column(df_sum,
                                                   risk="risk",
                                                   lower=0,
                                                   upper=1,
                                                   fml=".2f"
                                                   )
    return df_sum

