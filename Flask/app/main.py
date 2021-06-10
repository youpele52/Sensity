from flask import Flask, request, jsonify
from torch_utils import get_prediction, user_input_to_tensor, imshow

from flask import Flask, render_template
from PIL import Image
import base64
import io


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        text = data['input']
        # text = request.data()  # request.form['text']
        processed_text = '_'.join(str(text).split(' ')).lower()
        # print(processed_text)
        # return processed_text
        # return
        try:
            predicted_img = get_prediction(processed_text)
            save_gen_img = imshow(inp=predicted_img, title=processed_text)
            return jsonify({'response': 'Successfully generated a ' + processed_text})

        except ValueError as err:
            return jsonify({'error': 'error during conversion or predicition'}, {'error_message': err})

    return jsonify({'FashionMNIST': 10})


@app.route('/post', methods=['POST'])
def post_route():
    if request.method == 'POST':
        data = request.get_json(force=True)
        data = data['input']
        print('Data Received: "{data}"'.format(data=data))
        return "Request Processed.\n"


# go to the app folder in the terminal and run the following
# run this for hot reloading during the development process
# export FLASK_APP=main.py
# export FLASK_ENV=developoment
# flask run
