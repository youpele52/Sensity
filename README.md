# Sensity

## Critic/Generator model training

### Getting started

#### Installing required packages

For building the model and training , Go to /Sensity folder and run

Go to terminal and the working folder where model.py, train.py etc., are located and run this
pip install -r requirements.txt

### Trainig the models

Run this in the folder where train.py is to train the models

python train.py

## Prediction using Flask curl post request

#### Installing required packages

In two terminals opened, go to /Flask folder ie /Sensity/Flask

Activate the virtual environment using this commands in both terminals

source venv/bin/activate

Then install the required packages using this command
pip install -r requirements.txt

### Running Flask

Although not necessary since there is .env file to set Flask environment, nevertheless

(In one of the terminals run the flask using this commands
export FLASK_APP=main.py
export FLASK_ENV=developoment)

Run Flask, type this

flask run

### curl post request

In the second terminal, run the following curl post request

curl localhost:5000/predict -d '{"input": "sneakers"}'

Check this folder /Flask/app/generated_images to view the generated image

To make another post request, change the “sneakers” to something else

Unfortunately, after successful image generation and the flask will stop running. To make another post request, start flask again “flask run”, the you can make another request.

Fashion_MNIST_classes = {
0: "T-shirt/Top",
1: "Trouser",
2: "Pullover",
3: "Dress",
4: "Coat",
5: "Sandal",
6: "Shirt",
7: "Sneaker",
8: "Bag",
9: "Ankle Boot"
}
