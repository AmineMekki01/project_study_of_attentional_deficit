import os
import mne
import matplotlib.pyplot as plt
import numpy as np

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


def plot_confusion_matrix(cm, classes, model_name, dir_name=path_to_plots):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add annotations to cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{dir_name}/{model_name}_confusion_matrix.png')
    final_path = f'{dir_name}/{model_name}_confusion_matrix.png'
    return final_path
