

def plot_figure4_example_frame_sequence(data_dir, dataset_name):
    pass


def plot_figure4_example_traces(data_dir, dataset_name):
    pass


def plot_figure4_outlier_detector_performance(data_dir, dataset_name):
    pass


def plot_figure4_venn_diagrams(data_dir, dataset_name):
    pass


def plot_figure3(data_dir, dataset_name):

    # plot sequences of frames
    plot_figure4_example_frame_sequence(data_dir=data_dir, dataset_name=dataset_name)

    # plot traces of predictions and error metrics with colored backgrounds
    plot_figure4_example_traces(data_dir=data_dir, dataset_name=dataset_name)

    # plot metric performance as outlier detector, but only for multiview datasets
    if dataset_name in ['mirror-mouse', 'mirror-fish']:
        plot_figure4_outlier_detector_performance(data_dir=data_dir, dataset_name=dataset_name)

    # plot venn diagrams of outlier detector overlaps
    plot_figure4_venn_diagrams(data_dir=data_dir, dataset_name=dataset_name)
