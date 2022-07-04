import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent
CORR_DIR = os.path.join(ROOT_DIR, 'correlation_table')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def show_heatmap():
    csv = pd.read_csv(os.path.join(CORR_DIR, 'df_corr.csv'))
    ax = sns.heatmap(csv)
    plt.title('Item correlation')
    plt.show()
