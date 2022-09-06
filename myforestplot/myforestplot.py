from typing import Union, Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels


@dataclass(repr=True)
class BaseForestplot():
    """Class for creating forestplot.

    Args:
        ratio: Ratio for text part and figure part.
        df: Engineered dataframe.
        hide_spines : Hide outlines of axes.  Takes "right","top","bottom","left" or these list.
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

    def __post_init__(self):
        self.df = self.df.reset_index(drop=True)

        self.create_y_index()
        self.figure_layout()

    def create_y_index(self):
        self.n_row = self.df.shape[0]
        self.y_index = np.array([-i for i in range(self.n_row)])
        self.ymax = np.max(self.y_index)
        self.ymin = np.min(self.y_index)

    def figure_layout(self):
        """Create figure layout for plotting"""
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
                 xerr_lower: str = "xerr_lower", 
                 xerr_upper: str = "xerr_upper", 
                 y_adj: float = 0,
                 errorbar_kwds: dict = None,
                 ref_kwds: dict = None):
        """

        Args:
            kwds: Passed to ax.errorbar function.
        """
        y_index = self.y_index + y_adj
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
                          y_index[cond],
                          xerr=self.df.loc[cond, [xerr_lower, xerr_upper]].T,
                          **errorbar_kwds
                          )
        cond = self.df[risk].isnull()
        self.df["ref"] = self.df[risk].mask(cond, 1).mask(~cond, np.nan)
        self.ax2.scatter(self.df["ref"], y_index, **ref_kwds)

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
                                       xmax_ax1: float=None, 
                                       kwds: dict = None):
        """Draw horizontal lines for seprating variables.

        Args:
            kwds: Passed to ax.axhline function.
        """
        cond = self.df["category"].duplicated()
        cate_y_index = -np.array(cond[cond==False].index)
        hlines = cate_y_index.copy() + 0.5
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
                      ):
        """Embed strings/values of one column with header.
        """
        y_index = self.y_index + y_adj
        if text_kwds is None:
            text_kwds = {}
        if header_kwds is None:
            header_kwds = {}
        self.ax1.text(x, y_header, header, ha="left", va="center",
                      fontsize=fontsize, **header_kwds)

        # Drop duplicated items
        ser = self.df[col]
        if duplicate_hide: 
            cond = ser.duplicated()
            ser = ser.mask(cond, "")
        if replace is not None:
            ser = ser.replace(replace)

        for y, text in zip(y_index, ser):
            self.ax1.text(x, y, text, ha="left", va="center",
                          fontsize=fontsize, **text_kwds)






