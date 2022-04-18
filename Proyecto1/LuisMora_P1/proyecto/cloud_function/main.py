import os
import tempfile

from google.cloud import storage, vision
from wand.image import Image

storage_client = storage.Client()
client = vision.ImageAnnotatorClient()

#Funcion main
def main(event, context):
    print("Evento-contexto", context)

    image_to_open = 'images/cara.jpg'

    with open(image_to_open, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    face_response = client.face_detection(image=image)

    for face_detection in face_response.face_annotations:
        d={
            'confidence': face_detection.detection_confidence,
            'joy': face_detection.joy_likelihood,
            'sorrow': face_detection.sorrow_likelihood,
            'surprise': face_detection.surprise_likelihood,
            'anger': face_detection.anger_likelihood
        }
        print(d)
