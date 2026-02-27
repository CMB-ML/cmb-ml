import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Figure 

class LossPlotExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="loss_plot")
        self.out_fig: Figure = self.assets_out["fig"]
        self.in_loss_csv = self.assets_in["loss_record"]
        # self.skip_n_values = 10

        self.fig_label = cfg.fig_model_name
        
        self.fig_type = cfg.get("fig_type", None)

    def execute(self):
        self.out_fig.handler.set_fig_types(self.fig_type)
        fig, ax = make_loss_plot(some_path=self.in_loss_csv.path,
                                 fig_label=self.fig_label)
        self.out_fig.write(fig=fig)

def make_loss_plot(some_path, fig_label, figsize=(10,6), fig=None, ax=None, ylim=None, sm_sigma2=None):
        df = pd.read_csv(some_path)
        #skipping the first few rows if they are not needed
        # df = df[df["Epoch"] >= self.skip_n_values]

        best_idx = df["Validation Loss"].idxmin()
        best_val_loss = df.loc[best_idx, "Validation Loss"]
        best_epoch = df.loc[best_idx, "Epoch"]

        if (fig is None and ax is not None) or (fig is not None and ax is None):
             raise ValueError("fig and ax must be provided together")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df["Epoch"], df["Training Loss"], label="Training Loss", color="tab:blue", alpha=0.6, linewidth=0.5)
        ax.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", color="tab:orange", alpha=0.8, linewidth=0.5)

        loss_txt = exp_as_ltx(best_val_loss)
        ax.annotate(
            f"Best validation loss\n${loss_txt}$ at epoch {best_epoch}",
            xy=(best_epoch, best_val_loss),
            xytext=(50, 1e6),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        train = df["Training Loss"].values
        val = df["Validation Loss"].values

        train_log = np.log(train)
        val_log = np.log(val)

        train_smoothed = gaussian_filter1d(train_log, sigma=5)
        val_smoothed   = gaussian_filter1d(val_log, sigma=5)

        train_end_loss_txt = exp_as_ltx(np.exp(train_smoothed[-1]))
        valid_end_loss_txt = exp_as_ltx(np.exp(val_smoothed[-1]))
        
        ax.plot(df["Epoch"], np.exp(train_smoothed), label="Smoothed Training Loss", linestyle="--", linewidth=2, color="tab:blue")
        ax.plot(df["Epoch"], np.exp(val_smoothed), label="Smoothed Validation Loss", linestyle="--", linewidth=2, color="tab:orange")

        if sm_sigma2 is not None:
            train_smoothed = gaussian_filter1d(train_log, sigma=sm_sigma2)
            val_smoothed   = gaussian_filter1d(val_log, sigma=sm_sigma2)

            ax.plot(df["Epoch"], np.exp(train_smoothed), 
                    label="Smoothed Training Loss", 
                    linestyle="--", linewidth=4, alpha=0.5, 
                    color="tab:blue")
            ax.plot(df["Epoch"], np.exp(val_smoothed), 
                    label="Smoothed Validation Loss", 
                    linestyle="--", linewidth=4, alpha=0.5, 
                    color="tab:orange")

            ax.annotate(
                f"Approx Train loss\n${train_end_loss_txt}$",
                xy=(df["Epoch"].iloc[-1], np.exp(train_smoothed[-1])),
                xytext=(150, 5e5),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
            ax.annotate(
                f"Approx Valid loss\n${valid_end_loss_txt}$",
                xy=(df["Epoch"].iloc[-1], np.exp(val_smoothed[-1])),
                xytext=(200, 1.5e6),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

        ax.set_yscale('log')

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training vs Validation Loss, {fig_label}")

        if ylim is not None:
            ax.set_ylim(*ylim)

        custom_legend = [
            Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2, label="Training Loss"),
            Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=2, label="Validation Loss")
        ]
        ax.legend(handles=custom_legend)
        ax.grid(True)
        return fig, ax

def exp_as_ltx(some_val):
    val_sci = "{:.1e}".format(some_val)
    base, exp = val_sci.split("e")
    exp = int(exp)
    return f"{base} \\times 10^{{{exp}}}" if exp != 0 else base
