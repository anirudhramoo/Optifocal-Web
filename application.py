from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from utils.droid_eyes_web import attn_detector
import base64

application = Flask(__name__)
#app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(application)

vals = ["horizontal", "vertical", "pitch", "yaw", "roll", "startpt", "endpt"]
capacity = 4
cameraids = {}
cameras = []
freeids = [i for i in range(capacity)]
id = 3

for _ in range(capacity):
    cameras.append(attn_detector())

@application.route('/')
def home():
    return render_template("index.html")

@application.route('/video')
def video():
    return render_template("video.html")

@socketio.on('connect')
def gen_key():
    global id

    print("CONNECTED TO:", request.sid, id)
    if id >= 0:
        cameraids[request.sid] = freeids[id]
        id -= 1

    else:
        print("HIGH LOAD DISCONNECTING\n\n\n\n")
        emit("variables", {"horizontal": -4}, to=request.sid)

@socketio.on('disconnect')
def del_key():
    print("DISCONNECTED FROM:", request.sid, cameraids[request.sid])

    global id
    id += 1
    freeids[id] = cameraids[request.sid]

    cameras[cameraids[request.sid]].reset()
    cameraids.pop(request.sid, None)


#Updates when there's a new frame
@socketio.on("newframe")
def loadframe(image):
    print("image received from: ", request.sid)
    jdata = cameras[cameraids[request.sid]].update(base64.b64decode(image))
    # print("JDATA:", jdata)
    jdict = {t: v for (t, v) in zip(vals, jdata)}
    jdict["cameraids"] = cameraids[request.sid]
    # print(jdict)
    emit("variables", jdict, to=request.sid)


if __name__ == '__main__':
    socketio.run(application)
    # socketio.run(application, host='127.0.0.1',port='5000',debug=True)
    #socketio.run(application, host='0.0.0.0',port='5000',debug=True) # For Docker
