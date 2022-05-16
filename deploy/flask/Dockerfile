# Pull Docker image from DockerHub; TensorFlow with Py3
FROM tensorflow/tensorflow:2.2.3-py3

# Working dir /app
WORKDIR /app

# Transfer from ./ to /app in the container
ADD ./ /app

# Install audio dependencies
RUN apt-get update && apt-get install -y python3.8 libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install Py3 packages and upgrade pip
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Start configured uWSGI
CMD ["uwsgi", "app.ini"]