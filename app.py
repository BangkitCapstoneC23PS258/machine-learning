from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

class_names = {'ayam_bakar': 0,
 'ayam_betutu': 1,
 'ayam_goreng': 2,
 'ayam_pop': 3,
 'bakso': 4,
 'bakwan': 5,
 'batagor': 6,
 'beberuk_terong': 7,
 'capcay': 8,
 'coto_makasar': 9,
 'dendeng_batokok': 10,
 'gado_gado': 11,
 'gudeg': 12,
 'gulai_ikan': 13,
 'gulai_tambusu': 14,
 'gulai_tunjang': 15,
 'kerak_telur': 16,
 'kue_dadar_gulung': 17,
 'mie_aceh': 18,
 'nasi_goreng': 19,
 'nasi_kuning': 20,
 'papeda': 21,
 'pempek': 22,
 'peuyeum': 23,
 'rawon': 24,
 'rendang': 25,
 'sate': 26,
 'soto': 27,
 'telur_balado': 28}

our_model = load_model('model.h5', compile = False)

def predict_class(model, images):
  for img in images:
    img = image.load_img("static/"+img, target_size=(150, 150))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)/255                                   
    pred = model.predict(img)
    index = np.argmax(pred)
    pred = list(class_names.keys())[list(class_names.values()).index(index)]
  return pred

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Open Source API"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)
		
		images = []
		for file_name in os.listdir('static'):
			images.append(file_name)
		
		result = predict_class(our_model, images)

	return render_template("index.html", prediction = result, img_path = img_path)

if __name__ =='__main__':
	app.debug = True
	app.run(debug = True)
