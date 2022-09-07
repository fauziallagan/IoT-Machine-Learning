from flask import Flask, render_template, Response
from FaceSmileEyeDetection import VideoCamera
from Recognizer import VideoRecognizer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/face_recognition')
def recognizer():
    return render_template('recognizer.html')

@app.route('/video_recognition')
def video_recognition():
    return Response(gen(VideoRecognizer()),
                    mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
