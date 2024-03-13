

def plot_figure5_example_session(data_dir, dataset_name, format='pdf'):
    pass


def plot_figure5_session_stats(data_dir, dataset_name, format='pdf'):
    pass


def plot_figure5(data_dir, dataset_name, format='pdf'):

    # plot frames overlaid with predictions, dataset-specific metrics, and PSTHs
    plot_figure5_example_session(data_dir=data_dir, dataset_name=dataset_name, format=format)

    # plot metric stats over sessions
    plot_figure5_session_stats(data_dir=data_dir, dataset_name=dataset_name, format=format)
