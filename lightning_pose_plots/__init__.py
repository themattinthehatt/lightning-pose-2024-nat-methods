__version__ = "1.0.0"


import seaborn as sns
colors_tab10 = sns.color_palette("tab10")
colors = [colors_tab10[4], colors_tab10[3], colors_tab10[1], colors_tab10[2], colors_tab10[0]]
model_order = ['dlc', 'baseline', 'context', 'semi-super', 'semi-super context']
model_colors = {model: colors[i] for i, model in enumerate(model_order)}
