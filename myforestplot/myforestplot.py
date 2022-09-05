from typing import Union, Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels



def statsmodels_fitting_result_dataframe(
    res,
    alpha: float = 0.05
    ) -> pd.DataFrame:
    """Create category and item columns from the statsmodels result.
    Categorical results are divided into original column name (category) and 
    its items (item).

    Args:
        res: statsmodels' fitting results.
        alpha: The significance level for the confidence interval.
    """
    df = np.exp(res.conf_int(alpha=0.05))
    df["risk"] = np.exp(res.params)

    cate = "category"
    df[cate] = np.nan
    rename_dic = {}
    for ind in df.index:
        if "[" in ind:
            s1, s2 = ind.split("[")
            rename_dic[ind] = s2[2:-1]
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
                       categorical: Dict[str, List[str]],
                       order: List[str] = None,
                       ) -> pd.DataFrame:
    """Sort category and item based on categorical values.

    Args:
        df_ : dataframe containing category and item.
        categorical: Dictionary containing column names and its order of items.
        order : if specified, category is ordered based on this variable.

    """
    if order is None:
        order = df_["category"].unique()

    df_sorted = pd.DataFrame()
    for c in order:
        cond = df_["category"] == c
        dfM = df_[cond]
        if c in categorical.keys():
            lis = categorical[c]
            sort_dic = {l:i for i,l in enumerate(lis)}
            dfM = dfM.sort_values(by="item",key=lambda x: x.replace(sort_dic))
        df_sorted = pd.concat((df_sorted, dfM), axis=0)
    return df_sorted


def statsmodels_pretty_result_dataframe(
    data: pd.DataFrame,
    res,
    categorical: Dict[str, List[str]],
    order: List[str] = None,
    fml: str = ".2f",
) -> pd.DataFrame:
    """Obtain pretty result dataframe from statsmodels results.
    Fitting coefficients are converted by np.exp.

    Args:
        data: original dataframe.
        res: statsmodels results.
        categorical: Dictionary containing column names and its order of items.
        order : if specified, category is ordered based on this variable.
        fml: formula for f string of pretty risk.
    """
    if res.nobs != data.shape[0]:
        raise Exception(("Some observations were dropped when fitted, "
                         "check number of observations"
                        ))
    df_res = statsmodels_fitting_result_dataframe(res, alpha=0.05)
    df_nobs = count_category_frequency(data, categorical.keys())
    df_sum = pd.merge(df_res, df_nobs, 
                      on=["category", "item"], 
                      validate="1:1", 
                      how="outer")
    df_sum = sort_category_item(df_sum, categorical, order=order)
    df_sum["risk_pretty"] = add_pretty_risk_column(df_sum,
                                                   risk="risk",
                                                   lower=0,
                                                   upper=1,
                                                   fml=".2f"
                                                   )
    return df_sum


class Forestplot():
    def __init__(self, 
                 df: pd.DataFrame,
                 ratio: Tuple[float, float] = (8,3), 
                 figsize: Tuple[float, float] = (5,3),
                 dpi: int = 150,
                 hide_spines: Union[List[str],str] = ["left", "top", "right"]
                 ):
        """Class for creating forestplot.

        Args:
            ratio: Ratio for text part and figure part.
            df: Engineered dataframe.
            hide_spines : Hide outlines of axes.  Takes "right","top","bottom","left" or these list.
        """
        self.df = df.reset_index(drop=True)

        # Figure layout.
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(1, sum(ratio))

        ss1 = gs.new_subplotspec((0, 0), colspan=ratio[0])
        ss2 = gs.new_subplotspec((0, ratio[0]), colspan=ratio[1])

        self.ax1 = plt.subplot(ss1)
        self.ax2 = plt.subplot(ss2, sharey=self.ax1)

        # Hide spines.
        self.ax1.set_axis_off()
        self.ax2.spines[hide_spines].set_visible(False)
        self.ax2.set_yticks([])
        self.ax2.set_yticklabels([])

        # set attributes.
        self.n_row = df.shape[0]
        self.y_index = np.array([-i for i in range(self.n_row)])
        self.ymax = np.max(self.y_index)
        self.ymin = np.min(self.y_index)

        # Create empty field for text part.
        self.ax1.set_xlim([0,1])
        self.ax1.scatter([0, 0, 1, 1], [self.ymin, 0, self.ymin, 0], color="white")

    def errorbar(self, 
                 risk: str = "risk", 
                 xerr_lower: str = "xerr_lower", 
                 xerr_upper: str = "xerr_upper", 
                 errorbar_kwds: dict = None,
                 ref_kwds: dict = None):
        """

        Args:
            kwds: Passed to ax.errorbar function.
        """
        if errorbar_kwds is None:
            errorbar_kwds = dict(fmt="o", 
                                 capsize=5, 
                                 markeredgecolor="black",  
                                 ecolor="black", 
                                 color='white'
                                 )
        if ref_kwds is None:
            ref_kwds = dict(marker="s", s=20, color="black")
        cond = self.df[risk].notnull()
        self.ax2.errorbar(self.df.loc[cond, risk],
                          self.y_index[cond],
                          xerr=self.df.loc[cond, [xerr_lower, xerr_upper]].T,
                          **errorbar_kwds
                          )
        cond = self.df[risk].isnull()
        self.df["ref"] = self.df[risk].mask(cond, 1).mask(~cond, np.nan)
        self.ax2.scatter(self.df["ref"], self.y_index, **ref_kwds)

    def draw_horizontal_line(self,
                             y: float,
                             xmin_ax1: float=0, 
                             xmax_ax1: float=1.1, 
                             kwds: dict = None
                             ):
        """Draw horizontal line.
        """
        if kwds is None:
            kwds = dict(lw=1, ls="-", color="black")
        xmin, xmax = self.ax2.get_xlim()
        diff = xmax - xmin
        xmin = xmin - diff*0.1
        xmax = xmax + diff*0.1
        self.ax2.axhline(y=y, xmin=xmin, xmax=xmax, **kwds)
        self.ax1.axhline(y=y, xmin=xmin_ax1, xmax=xmax_ax1, 
                         zorder=-10, clip_on=False, 
                         **kwds)

    def horizontal_variable_separators(self, 
                                       xmin_ax1: float=0, 
                                       xmax_ax1: float=1.1, 
                                       kwds: dict = None):
        """Draw horizontal lines for seprating variables.

        Args:
            kwds: Passed to ax.axhline function.
        """
        cond = self.df["category"].duplicated()
        cate_y_index = -np.array(cond[cond==False].index)
        hlines = cate_y_index.copy() + 0.5
        xmin, xmax = self.ax2.get_xlim()
        for y in hlines:
            self.draw_horizontal_line(y=y, 
                                      xmin_ax1=xmin_ax1, 
                                      xmax_ax1=xmax_ax1, 
                                      kwds=kwds)

    def embed_strings(self, 
                      col: str, 
                      x: float, 
                      header: str, 
                      fontsize: int = 12,
                      y_header: float = 0.8,
                      text_kwds: Optional[dict] = None,
                      header_kwds: Optional[dict] = None,
                      duplicate_hide: bool = False,
                      replace: Optional[dict] = None,
                      ):
        """Embed strings/values of one column with header.
        """
        if text_kwds is None:
            text_kwds = {}
        if header_kwds is None:
            header_kwds = {}
        self.ax1.text(x, y_header, header, ha="left", va="center",
                      fontsize=fontsize, **header_kwds)

        ser = self.df[col]
        if duplicate_hide: 
            cond = ser.duplicated()
            ser = ser.mask(cond, "")
        if replace is not None:
            ser = ser.replace(replace)

        for y, text in zip(self.y_index, ser):
            self.ax1.text(x, y, text, ha="left", va="center",
                          fontsize=fontsize, **text_kwds)






