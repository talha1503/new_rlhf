import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import streamlit as st

sns.set_style("whitegrid")
plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black"})
# plt.style.use("ggplot")
# plt.rcParams['axes.edgecolor'] = "#141414"
# plt.rcParams['xtick.color'] = "#141414"
# plt.rcParams['ytick.color'] = "#141414"

@st.cache_resource
def generate_plots(filepath, saved_plot_dir):
    df = pd.read_csv(filepath)
    for summary_statistic in [
        "acceptance rate",
        "default rate",
        "average credit score",
    ]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(summary_statistic.capitalize(), fontsize=40, labelpad=15)
        ax.set_xlabel("Timestep", fontsize=40, labelpad=15)
        ax.plot(
            range(df.shape[0]),
            df[summary_statistic.replace(" ", "_") + "-group_1"],
            label="Group 1",
            linewidth=3,
        )
        ax.plot(
            range(df.shape[0]),
            df[summary_statistic.replace(" ", "_") + "-group_2"],
            label="Group 2",
            linewidth=3,
        )
        ax.legend(fontsize=30, labelcolor="black", loc="upper right", bbox_to_anchor=(1.3,1))
        ax.tick_params(labelsize=30, colors="black")

        os.makedirs(saved_plot_dir, exist_ok=True)
        fig.patch.set_alpha(0.0)
        fig.savefig(
            os.path.join(saved_plot_dir, summary_statistic + ".png"),
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join('./', summary_statistic + ".png"),
            bbox_inches="tight",
        )
