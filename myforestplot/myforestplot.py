from typing import Union, Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels

import myforestplot.vis_utils as vis_utils


@dataclass(repr=True)
class BaseForestPlot():
    """Class for creating a forest plot.

    Args:
        ratio: Ratio for text part and figure part.
        df: Engineered dataframe.
        hide_spines : Hide outlines of axes.  Takes "right","top","bottom","left" or these list.
        vertical_align: Align categorical names above items. It requires dataframe to have 
            "category" and "item" column names.
    """
    df: pd.DataFrame
    ratio: Tuple[float, float] = (8,3)
    figsize: Tuple[float, float] = (5,3)
    dpi: int = 150
    hide_spines: List[str] = field(default_factory=lambda: ["left", "top", "right"]) 
    yticks_show: bool = False
    yticklabels_show: bool = False
    xticks_show: bool = True
    text_axis_off: bool = True
    vertical_align: bool = False

    def __post_init__(self):
        self.df = self.df.reset_index(drop=True)

        self.create_y_index()
        self.figure_layout()

    def create_y_index(self):
        self.n_item = self.df.shape[0]
        if self.vertical_align:
            if "category" not in self.df.columns:
                raise Exception("Need 'category' column for df variable.")
            if "item" not in self.df.columns:
                raise Exception("Need 'item' column for df variable.")

            self.y_index_cate, self.y_index = (
                vis_utils.obtain_indexes_from_category_item(
                    self.df["category"],
                    self.df["item"]
                    )
                )
        else:
            self.y_index = np.array([-i for i in range(self.n_item)])
            cond = self.df["category"].duplicated()
            self.y_index_cate = -np.array(cond[cond==False].index)

        self.ymax = np.max(self.y_index_cate)
        self.ymin = np.min(self.y_index)

    def figure_layout(self):
        """Create figure layout for plotting"""
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"

        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(1, sum(self.ratio))

        ss1 = gs.new_subplotspec((0, 0), colspan=self.ratio[0])
        ss2 = gs.new_subplotspec((0, self.ratio[0]), colspan=self.ratio[1])

        self.ax1 = plt.subplot(ss1)
        self.ax2 = plt.subplot(ss2, sharey=self.ax1)

        # Hide spines.
        if self.text_axis_off:
            self.ax1.set_axis_off()
        self.ax2.spines[self.hide_spines].set_visible(False)
        # Show or hide ticks
        if not self.xticks_show:
            self.ax2.set_xticks([])
        if self.yticks_show:
            self.ax2.set_yticks(self.y_index)
        else:
            self.ax2.set_yticks([])
        if not self.yticklabels_show:
            self.ax2.set_yticklabels([])

        # Figure restriction.
        self.ax2.set_ylim([self.ymin - 0.5, 0.5])

        # Create empty field for text part.
        self.ax1.set_xlim([0,1])
        self.ax1.scatter([0, 0, 1, 1], [self.ymin, 0, self.ymin, 0], color="white")

    def errorbar(self, 
                 risk: str = "risk", 
                 lower: Union[str, int] = 0, 
                 upper: Union[str, int] = 1, 
                 y_adj: float = 0,
                 errorbar_kwds: Optional[dict] = None,
                 ref_kwds: Optional[dict] = None,
                 df: Optional[pd.DataFrame] = None,
                 errorbar_color: Optional[str] = None,
                 ref_color: Optional[str] = None,
                 label: Optional[str] = None,
                 log_scale: bool = False,
                 ):
        """

        Args:
            risk: Column name for risk.
            lower: Column name for lower confidence interval.
            upper: Column name for upper confidence interval.
            y_adj: For this value, plotting is moved. 
            errorbar_kwds: Passed to ax.errorbar function.
            ref_kwds: Passed to ax.scatter function.
            df: Dataframe for another result.
            errorbar_color: If specified, ecolor and coloer in erorrbar_kwds is 
                changed to this value.
            ref_color: If specified, ecolor and coloer in ref_kwds is 
                changed to this value.
            label: Label for stratified drawings. Passed to ax.errorbar.
            log_scale: Plot risk in log scale (np.log).
        """
        if df is None:
            df = self.df
        if errorbar_color is not None:
            errorbar_kwds["ecolor"] = errorbar_color
            errorbar_kwds["color"] = errorbar_color
        if ref_color is not None:
            ref_kwds["color"] = ref_color

        vis_utils.errorbar_forestplot(
            ax=self.ax2, 
            y_index=self.y_index,
            df=df,
            risk=risk,
            lower=lower,
            upper=upper,
            y_adj=y_adj,
            errorbar_kwds=errorbar_kwds,
            ref_kwds=ref_kwds,
            errorbar_color=errorbar_color,
            ref_color=ref_color,
            label=label,
            log_scale=log_scale,
        )

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
        xmin = -1_000_000
        xmax =  1_000_000
        self.ax2.axhline(y=y, xmin=xmin, xmax=xmax, **kwds)
        self.ax1.axhline(y=y, xmin=xmin_ax1, xmax=xmax_ax1, 
                         zorder=-10, clip_on=False, 
                         **kwds)

    def horizontal_variable_separators(self, 
                                       xmin_ax1: float=0, 
                                       xmax_ax1: float=None, 
                                       kwds: dict = None):
        """Draw horizontal lines for seprating variables.

        Args:
            kwds: Passed to ax.axhline function.
        """
        hlines = self.y_index_cate.copy() + 0.5
        xmin, xmax = self.ax2.get_xlim()

        # to cover figure part.
        if xmax_ax1 is None:
            xmax_ax1 = 1 + self.ratio[1]/self.ratio[0]*1.1
        for y in hlines:
            self.draw_horizontal_line(y=y, 
                                      xmin_ax1=xmin_ax1, 
                                      xmax_ax1=xmax_ax1, 
                                      kwds=kwds)

    def embed_strings(self, 
                      col: str, 
                      x: float, 
                      header: str, 
                      fontsize: int = None,
                      y_header: float = 1.0,
                      y_adj : float = 0.0,
                      text_kwds: Optional[dict] = None,
                      header_kwds: Optional[dict] = None,
                      duplicate_hide: bool = False,
                      replace: Optional[dict] = None,
                      df: Optional[pd.DataFrame] = None,
                      ):
        """Embed strings/values of one column with header.

        Args:
            col: Column name for text.
            x: x axis value of text position, ranging from 0 to 1.
            df: Dataframe for another result.
        """
        if df is None:
            df = self.df
        ser = df[col]
        # Drop duplicated items
        if duplicate_hide: 
            cond = ser.duplicated()
            ser = ser.mask(cond, "")

        y_index = self.y_index + y_adj

        vis_utils.embed_strings_forestplot(
            ax=self.ax1,
            ser=ser,
            y_index=y_index,
            x=x,
            header=header,
            fontsize=fontsize,
            y_header=y_header,
            text_kwds=text_kwds,
            header_kwds=header_kwds,
            replace=replace
        )

    def embed_cate_strings(self,
                           col: str,
                           x: float,
                           header: str,
                           fontsize: int = None,
                           y_header: float = 1.0,
                           y_adj : float = 0.0,
                           text_kwds: Optional[dict] = None,
                           header_kwds: Optional[dict] = None,
                           replace: Optional[dict] = None,
                           df: Optional[pd.DataFrame] = None,
                           ):
        """Embed category values on vertically aligned positions.
        The position of strings become different only if self.vertical_align == True.
        """
        if df is None:
            df = self.df
        ser = df[col].drop_duplicates()

        y_index = self.y_index_cate + y_adj

        vis_utils.embed_strings_forestplot(
            ax=self.ax1,
            ser=ser,
            y_index=y_index,
            x=x,
            header=header,
            fontsize=fontsize,
            y_header=y_header,
            text_kwds=text_kwds,
            header_kwds=header_kwds,
            replace=replace
        )


@dataclass(repr=True)
class MultiForestPlot():
    """

    Args:
        df: Result dataframe.
        ratio: Number of axes field and size ratio of these axes.
            self.axd contains axes of which index starts from 1. 
        fig_ax_index: If specified, x ticks and x labels are left.
        figsize: Figure size.
        hide_spines : Hide outlines of axes.  Takes "right","top","bottom","left" or these list.
        vertical_align: Align categorical names above items. It requires dataframe to have 
            "category" and "item" column names.
    """
    df: pd.DataFrame
    ratio: Tuple[float, float] = (8,3)
    fig_ax_index: Optional[List[int]] = None
    figsize: Tuple[float, float] = (5,3)
    yticks_show: bool = False
    yticklabels_show: bool = False
    xticks_show: bool = True
    hide_spines: List[str] = field(default_factory=lambda: ["left", "top", "right"]) 
    dpi: int = 150
    vertical_align: bool = False


    def __post_init__(self):
        self.df = self.df.reset_index(drop=True)

        self.create_y_index()
        self.figure_layout()

    def create_y_index(self):
        self.n_item = self.df.shape[0]
        if self.vertical_align:
            if "category" not in self.df.columns:
                raise Exception("Need 'category' column for df variable.")
            if "item" not in self.df.columns:
                raise Exception("Need 'item' column for df variable.")

            self.y_index_cate, self.y_index = (
                vis_utils.obtain_indexes_from_category_item(
                    self.df["category"],
                    self.df["item"]
                    )
                )
        else:
            self.y_index = np.array([-i for i in range(self.n_item)])
            cond = self.df["category"].duplicated()
            self.y_index_cate = -np.array(cond[cond==False].index)

        self.ymax = np.max(self.y_index_cate)
        self.ymin = np.min(self.y_index)

    def figure_layout(self):
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"

        mosaic = [[]]
        for i,r_num in enumerate(self.ratio):
            for r in range(r_num):
                mosaic[0].append(i+1)
        self.fig = plt.figure(constrained_layout=False)
        self.axd = self.fig.subplot_mosaic(mosaic, sharex=False, sharey=True)

        # Figure restriction.
        ax = self.axd[1]
        ax.set_ylim([self.ymin - 0.5, 0.5])


        if self.fig_ax_index is None:
            self.fig_ax_index = []
        for i in range(len(self.ratio)):
            ax_index = i + 1
            ax = self.axd[ax_index]

            if ax_index in self.fig_ax_index:
                # Show or hide ticks
                diff = set(self.hide_spines) - set(ax.spines.keys())
                if len(diff) > 0:
                    raise Exception(("hide_spines should only take each one of"
                                     "'bottom', 'left', 'right', 'top'."))
                ax.spines[self.hide_spines].set_visible(False)
                if not self.xticks_show:
                    ax.set_xticks([])
                if self.yticks_show:
                    ax.set_yticks(self.y_index)
                else:
                    ax.set_yticks([])
                if not self.yticklabels_show:
                    ax.set_yticklabels([])
            else:
                # For text field.
                ax.set_xlim([0,1])
                ax.scatter([0, 0, 1, 1], [self.ymin, 0, self.ymin, 0], color="white")
                ax.set_axis_off()

    def errorbar(self, 
                 index: int,
                 risk: str = "risk", 
                 lower: Union[str, int] = 0, 
                 upper: Union[str, int] = 1, 
                 y_adj: float = 0,
                 errorbar_kwds: Optional[dict] = None,
                 ref_kwds: Optional[dict] = None,
                 df: Optional[pd.DataFrame] = None,
                 errorbar_color: Optional[str] = None,
                 ref_color: Optional[str] = None,
                 label: Optional[str] = None,
                 log_scale: bool = False,
                 ):
        """

        Args:
            index: axis's index.
            risk: Column name for risk.
            lower: Column name for lower confidence interval.
            upper: Column name for upper confidence interval.
            y_adj: For this value, plotting is moved. 
            errorbar_kwds: Passed to ax.errorbar function.
            ref_kwds: Passed to ax.scatter function.
            df: Dataframe for another result.
            errorbar_color: If specified, ecolor and coloer in erorrbar_kwds is 
                changed to this value.
            ref_color: If specified, ecolor and coloer in ref_kwds is 
                changed to this value.
            label: Label for stratified drawings. Passed to ax.errorbar.
            log_scale: Plot risk in log scale (np.log).
        """
        ax = self.axd[index]
        if df is None:
            df = self.df
        if errorbar_color is not None:
            errorbar_kwds["ecolor"] = errorbar_color
            errorbar_kwds["color"] = errorbar_color
        if ref_color is not None:
            ref_kwds["color"] = ref_color

        vis_utils.errorbar_forestplot(
            ax=ax, 
            y_index=self.y_index,
            df=df,
            risk=risk,
            lower=lower,
            upper=upper,
            y_adj=y_adj,
            errorbar_kwds=errorbar_kwds,
            ref_kwds=ref_kwds,
            errorbar_color=errorbar_color,
            ref_color=ref_color,
            label=label,
            log_scale=log_scale,
        )

    def embed_strings(self, 
                      index: int,
                      col: str, 
                      x: float, 
                      header: str, 
                      fontsize: int = None,
                      y_header: float = 1.0,
                      y_adj : float = 0.0,
                      text_kwds: Optional[dict] = None,
                      header_kwds: Optional[dict] = None,
                      duplicate_hide: bool = False,
                      replace: Optional[dict] = None,
                      df: Optional[pd.DataFrame] = None,
                      ):
        """Embed strings/values of one column with header.

        Args:
            col: Column name for text.
            x: x axis value of text position, ranging from 0 to 1.
            df: Dataframe for another result.
        """
        ax = self.axd[index]
        if df is None:
            df = self.df
        ser = df[col]
        # Drop duplicated items
        if duplicate_hide: 
            cond = ser.duplicated()
            ser = ser.mask(cond, "")

        y_index = self.y_index + y_adj

        vis_utils.embed_strings_forestplot(
            ax=ax,
            ser=ser,
            y_index=y_index,
            x=x,
            header=header,
            fontsize=fontsize,
            y_header=y_header,
            text_kwds=text_kwds,
            header_kwds=header_kwds,
            replace=replace
        )

    def embed_cate_strings(self,
                           index: int,
                           col: str,
                           x: float,
                           header: str,
                           fontsize: int = None,
                           y_header: float = 1.0,
                           y_adj : float = 0.0,
                           text_kwds: Optional[dict] = None,
                           header_kwds: Optional[dict] = None,
                           replace: Optional[dict] = None,
                           df: Optional[pd.DataFrame] = None,
                           ):
        """Embed category values on vertically aligned positions.
        The position of strings become different only if self.vertical_align == True.
        """
        ax = self.axd[index]
        if df is None:
            df = self.df
        ser = df[col].drop_duplicates()

        y_index = self.y_index_cate + y_adj

        vis_utils.embed_strings_forestplot(
            ax=ax,
            ser=ser,
            y_index=y_index,
            x=x,
            header=header,
            fontsize=fontsize,
            y_header=y_header,
            text_kwds=text_kwds,
            header_kwds=header_kwds,
            replace=replace
        )
