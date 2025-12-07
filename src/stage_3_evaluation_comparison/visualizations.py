import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to plot bar chart for percentage changes
def plot_bar(df, title, value_col="% Change (NN vs XGB)"):
    plt.figure(figsize=(12,6))
    bars = plt.bar(df["Metric"], df[value_col], color='skyblue')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Percentage Change (%)")
    plt.title(title)
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.annotate(f'{height:+.2f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0,3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()
