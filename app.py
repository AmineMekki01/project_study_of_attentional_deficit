import os
from flask import send_file
from flask import Flask, request, jsonify, render_template, send_from_directory
from typing import Dict, Tuple, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

from src.utils.plotting import *
from src.components.feature_selection import feature_selection_RFC, feature_selection_LDA, feature_selection_RFE
from src.pipeline.train_pipeline import training_pipeline
from src.pipeline.test_pipeline import testing_pipeline


from sklearn.svm import SVC
app = Flask(__name__)

UPLOAD_FOLDER = './artifacts/data/raw/AEP/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

path_to_plots = './artifacts/plots'

feature_selection_methods: Dict[str, Callable] = {
    "RFC": feature_selection_RFC,
    "LDA": feature_selection_LDA,
    "RFE": feature_selection_RFE
}

# Models for training
models: Dict[str, Callable] = {
    "Random Forest": RandomForestClassifier(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Support Vector Machine": SVC(probability=True)
}


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        feature_selection_method = request.form['featureSelection']
        model_to_use = request.form['model']
        cross_validation_method = request.form['CrossValidationMethod']

        chosen_FS_methods = {
            feature_selection_method: feature_selection_methods[feature_selection_method]}

        chosen_model = {
            model_to_use: models[request.form['model']]
        }

        set_file = request.files['set_file']
        fdt_file = request.files['fdt_file']

        if set_file.filename != '' and fdt_file.filename != '':
            set_path = f"./artifacts/uploaded_data/{set_file.filename}"
            fdt_path = f"./artifacts/uploaded_data/{fdt_file.filename}"

            set_file.save(set_path)
            fdt_file.save(fdt_path)

            metrics_df = training_pipeline(
                set_path, chosen_FS_methods, chosen_model, cross_validation_method)

            print("Metrics DF:", metrics_df)

            epochs = read_eeg_data(set_path)

            filtered_epochs = filter_eeg_data(epochs)

            psd_path = plot_psd(filtered_epochs, path_to_plots)

            raw_data_path = plot_raw_data(filtered_epochs, path_to_plots)

            plot_paths = {
                'psd_path': psd_path,
                'raw_data_path': raw_data_path,
            }

            return jsonify({'metrics': metrics_df.to_dict(orient='records'), **plot_paths})

    elif request.method == 'GET':
        return render_template('trainingPage.html')
    else:
        abort(405)


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        feature_selection_method = request.form['featureSelection']
        model_to_use = request.form['model']

        set_file = request.files['set_file']
        fdt_file = request.files['fdt_file']

        if set_file.filename != '' and fdt_file.filename != '':
            set_path = f"./artifacts/uploaded_data/{set_file.filename}"
            fdt_path = f"./artifacts/uploaded_data/{fdt_file.filename}"

            set_file.save(set_path)
            fdt_file.save(fdt_path)

            test_metrics_df = testing_pipeline(
                set_path, model_to_use, feature_selection_method)

            print("Test Metrics DF:", test_metrics_df)

            epochs = read_eeg_data(set_path)

            filtered_epochs = filter_eeg_data(epochs)

            psd_path = plot_psd(filtered_epochs, path_to_plots)

            raw_data_path = plot_raw_data(filtered_epochs, path_to_plots)

            plot_paths = {
                'psd_path': psd_path,
                'raw_data_path': raw_data_path,
            }

            return jsonify({'metrics': test_metrics_df.to_dict(orient='records'), **plot_paths})

    elif request.method == 'GET':
        return render_template('testPage.html')
    else:
        abort(405)


@app.errorhandler(405)
def method_not_allowed(e):
    return "Method not allowed. Please use the correct HTTP verb.", 405


@app.route('/artifacts/plots/<filename>')
def serve_plot(filename):
    return send_file(f'.\\artifacts\\plots\\{filename}', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)