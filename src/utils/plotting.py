import os
import mne

path_to_plots = './../../artifacts/plots'


def init_plot_directory(dir_name=path_to_plots):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def read_eeg_data(path):
    import mne
    epochs = mne.io.read_epochs_eeglab(path)
    return epochs


def filter_eeg_data(epochs, l_freq=1, h_freq=30):
    epochs.filter(l_freq, h_freq)
    return epochs


def plot_psd(epochs, dir_name=path_to_plots):
    plot_path = f"{dir_name}/psd.png"
    epochs.plot_psd(show=False).savefig(plot_path)
    return plot_path


def plot_raw_data(epochs, dir_name=path_to_plots):
    plot_path = f"{dir_name}/raw_data.png"
    epochs.plot(n_epochs=4, show=False).savefig(plot_path)
    return plot_path


def plot_sensor_locations(epochs, dir_name=path_to_plots):
    plot_2d_path = f"{dir_name}/sensor_2d.png"
    plot_3d_path = f"{dir_name}/sensor_3d.png"
    epochs.plot_sensors(kind='topomap', show=False).savefig(plot_2d_path)
    epochs.plot_sensors(kind='3d', show=False).savefig(plot_3d_path)
    return plot_2d_path, plot_3d_path
