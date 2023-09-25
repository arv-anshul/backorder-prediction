import pandas as pd
from flask import Flask, jsonify, render_template, request

from backorder import pipeline
from backorder.entity import DataIngestionConfig

app = Flask(__name__)
ingestion_config = DataIngestionConfig()
training = pipeline.Training()
prediction = pipeline.Prediction()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train_model', methods=['POST'])
def train_model():
    training.initiate()
    return jsonify({'message': 'Model Training Completed!'})


@app.route('/predict')
def predict():
    return render_template(
        'predict.html',
        num_cols=ingestion_config.num_cols,
        cat_cols=ingestion_config.cat_cols,
    )


@app.route('/one_prediction', methods=['POST'])
def one_prediction():
    form_data = dict(request.form)
    df = pd.DataFrame([form_data.values()], columns=list(form_data.keys()))
    prediction = pipeline.Prediction.one_prediction(df)
    return jsonify(list(prediction.T.to_dict().values())[0])


@app.route('/batch_prediction', methods=['POST'])
def batch_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        df = pd.read_csv(file.stream)
        prediction_fp = prediction.batch_prediction(df)
        return {
            'message': 'Prediction Completed!',
            'prediction_path': prediction_fp.absolute().as_uri(),
        }
    except Exception as e:
        return jsonify({'error': f"Error occurred: {e}"})


if __name__ == '__main__':
    app.run(port=8501)
