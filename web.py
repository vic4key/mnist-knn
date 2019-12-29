from os import environ
from flask import Flask, request
from flask_cors import CORS, cross_origin

from PyVutils import Cv

from utils import preprocess, data_uri_to_cv2_img
from main import predict_img

app = Flask(__name__)

cors = CORS(app) # Allow Cross-Origin Resource Sharing

@app.route("/")
def api_root(): return app.send_static_file("interface.html")

@app.route("/post-data-url", methods=["POST"])
@cross_origin()
def api_predict_from_dataurl():

    imgstring = request.form.get("data")
    img = preprocess(data_uri_to_cv2_img(imgstring))
    img = Cv.CombineRGB(img, Cv.CN_BLUE)
    img = Cv.Invert(img)

    result = predict_img(img, False)

    print(F"Prediction requested and returned {result}")

    return result

def run(): return app.run(debug=False, port=environ.get("PORT", 5000), host="127.0.0.1")