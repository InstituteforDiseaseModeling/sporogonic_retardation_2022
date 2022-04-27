import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'font.size': 16})

exp_name = 'sporozoite_reduction'
data_dir = os.path.join(os.path.expanduser('~'), 'Github', 'emodpy-vector_genetics', 'data', "Niki")
fig_dir = os.path.join(os.path.expanduser('~'), 'Github', 'emodpy-vector_genetics', 'figures', "Niki")
os.makedirs(fig_dir, exist_ok=True)


def new_colormap():
    bottom = pl.cm.get_cmap('Oranges', 128)
    top = pl.cm.get_cmap('Blues_r', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = mpl.colors.ListedColormap(newcolors, name='OrangeBlue')

    colormap = newcmp

    return colormap

df_incidence = pd.DataFrame()
for f in os.listdir(data_dir):
    if 'clinical_cases' in f:
        dftemp = pd.read_csv(os.path.join(data_dir, f))
        if 'baseline' in f:
            dftemp['Baseline'] = [1] * len(dftemp)
        else:
            dftemp['Baseline'] = [0] * len(dftemp)
        df_incidence = pd.concat([df_incidence, dftemp])

df_elimination = df_incidence[df_incidence['Baseline'] == 0]
df_incidence = df_incidence[df_incidence['Baseline'] == 1]

larval_capacity = [8.15, 8.45, 8.65]
r1_resistance = list(df_elimination['R1_resistance'].unique())

fig, axs = plt.subplots(2, 5, figsize=(30, 10))
cbar_ax = fig.add_axes([.92, .53, .01, .35])
cbar_ax2 = fig.add_axes([.92, .11, .01, .35])
df_inc = df_incidence[df_incidence['Larval_Capacity'] == 8.65]
for k, l in enumerate(r1_resistance):
    df_elim = df_elimination[df_elimination['R1_resistance'] == l]
    incidence_reduction = (1 - np.array(df_elim['New Clinical Cases']) / np.array(df_inc['New Clinical Cases'])) * 100
    df_elim['Incidence_reduction'] = incidence_reduction

    # Plot incidence reduction
    index = df_elim['Transmission_To_Human'].unique()
    cols = df_elim['Infected_Progress'].unique()
    B = np.reshape(np.array(df_elim['Incidence_reduction']), (-1, 5))
    labels = np.reshape(np.round(np.array(df_elim['Incidence_reduction'])), (-1, 5))
    df_plot = pd.DataFrame(B, columns=cols, index=index)

    sns.heatmap(B, ax=axs[1, k], cmap=new_colormap(), annot=labels, annot_kws={'fontsize': 16}, fmt='0.2g',
                vmin=0, vmax=100, cbar_ax=None if k else cbar_ax2, cbar=k == 0,
                cbar_kws={'ticks': [0, 25, 50, 75, 100]})
    axs[1, k].invert_xaxis()

    # Plot elimination
    index = df_elim['Transmission_To_Human'].unique()
    cols = df_elim['Infected_Progress'].unique()
    B = np.reshape(np.array(df_elim['Elimination'] * 100), (-1, 5))
    labels = np.reshape(np.round(np.array(df_elim['Elimination'] * 100)), (-1, 5))
    df_plot = pd.DataFrame(B, columns=cols, index=index)
    sns.heatmap(B, ax=axs[0, k], cmap="YlGnBu", annot=labels, annot_kws={'fontsize': 16}, fmt='0.4g',
                vmin=0, vmax=100, cbar_ax=None if k else cbar_ax, cbar=k == 0,
                cbar_kws={'ticks': [0, 25, 50, 75, 100]})
    # axs[0, k].invert_yaxis()
    axs[0, k].invert_xaxis()

    for yax in range(1, 5):
        axs[0, yax].set_yticks([])
        axs[1, yax].set_yticks([])
    axs[0, k].set_xticks([])
    xticklabels = ['%i%%' % (100 - i) for i in np.array(cols) * 100]
    yticklabels = ['%i%%' % (100 - i) for i in np.array(index) * 100]
    axs[1, k].set_xticklabels(xticklabels)
    axs[1, 0].set_yticklabels(yticklabels)
    axs[0, 0].set_yticklabels(yticklabels)
    axs[1, 0].set_ylabel('Sporozoite reduction')
    axs[0, 0].set_ylabel('Sporozoite reduction')
    axs[1, k].set_xlabel('Delay')

    if k == 0:
        ax2 = axs[0, len(r1_resistance) - 1].twinx()
        ax2.set_ylabel('Elimination\nprobability', color='b', rotation=0, labelpad=50,
                       fontdict={'size': 16})
    else:
        ax2 = axs[1, len(r1_resistance) - 1].twinx()
        ax2.set_ylabel('Incidence\nreduction', color='b', rotation=0, labelpad=50,
                       fontdict={'size': 16})
    ax2.set_yticks([])
    for _, spine in ax2.spines.items():
        spine.set_visible(False)

    axs[0, k].set_title('R1 resistance = %0.1f%%' % (l*100))

fig.tight_layout(rect=[0, 0, .92, 0.96])
plt.savefig(os.path.join(fig_dir, 'heatmap_mortality.pdf'))
plt.savefig(os.path.join(fig_dir, 'heatmap_mortality.png'))
