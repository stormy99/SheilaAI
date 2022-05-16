import librosa
import numpy as np
from tensorflow import keras

MODEL_PATH = "model.h5"  # trained model
NUM_SAMPLES_TO_CONSIDER = 22050  # 1 second


# Singleton design pattern
class _Keyword_Spotting_Service:
    model = None
    _mapping = [
        "down",
        "eight",
        "five",
        "four",
        "go",
        "left",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "two",
        "up",
        "yes",
        "zero",
    ]
    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # (# segments, # coefficients)

        # convert 2-Dimensional MFCC array into 4-Dimensional array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]  # (# samples, # segments, # coefficients, # channels)

        # make prediction
        predictions = self.model.predict(MFCCs)  # Prediction array [ [0.1, 0.7, 0.1, ...] ]
        predicted_index = np.argmax(predictions)  # Get array index value from the highest prediction
        predicted_keyword = self._mapping[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audiofile
        signal, sr = librosa.load(file_path)

        # ensure consistency in audiofile length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T


def Keyword_Spotting_Service():
    # ensure only 1 instance of Keyword Spotting Service
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    assert kss is kss1

    keyword = kss.predict("test/off.wav")
    print(f"Predicted Keyword: {keyword}")
