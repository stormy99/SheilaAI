import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service

"""
server
xyz.co.uk/predict

client -> POST request -> server -> return prediction to client
"""

# create a flask application
app = Flask(__name__)


# route incoming requests to an API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # get and temporarily save audiofile
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 256000))
    audio_file.save(file_name)

    # invoke keyword-spotting-service
    kss = Keyword_Spotting_Service()

    # predict
    predicted_keyword = kss.predict(file_name)

    # remove temporary audiofile
    os.remove(file_name)

    # return predicted keyword in JSON
    data = {"keyword": predicted_keyword}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)
