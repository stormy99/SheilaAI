import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIOFILE_PATH = "test/on.wav"

if __name__ == "__main__":

    audio_file = open(TEST_AUDIOFILE_PATH, "rb")
    values = {"file": (TEST_AUDIOFILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()
    
    print(f"Predicted Keyword is: {data['keyword']}")
