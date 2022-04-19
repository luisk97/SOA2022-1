# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=pointless-string-statement
from google.cloud import vision, storage # pylint: disable=import-error

storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

#Funcion main
def main(data, context):# pylint: disable=missing-function-docstring
    print("Evento-contexto", context)

    file_data = data

    file_name = file_data["name"]
    print(f"A ver {file_name}.")
    bucket_name = file_data["bucket"]

    #blob = storage_client.bucket(bucket_name).get_blob(file_name)
    blob_uri = f"gs://{bucket_name}/{file_name}"

    blob_source = vision.Image(source=vision.ImageSource(image_uri=blob_uri))


    print(f"Analyzing {file_name}.")

    """ image_to_open = 'images/cara.jpg'

    with open(image_to_open, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content) """

    face_response = vision_client.face_detection(image=blob_source)

    for face_detection in face_response.face_annotations:
        d={
            'confidence': face_detection.detection_confidence,
            'joy': face_detection.joy_likelihood,
            'sorrow': face_detection.sorrow_likelihood,
            'surprise': face_detection.surprise_likelihood,
            'anger': face_detection.anger_likelihood
        }
        print(d)
