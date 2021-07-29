import socket

import socketio
from time import sleep

# Data IO and Encoding-Decoding
from io import BytesIO
import base64

# Image Processing
from PIL import Image

import numpy as np

adv_patch_boxes = []

def send_img(ip, port, img):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = (ip, port)
    print('connecting to %s port %s' % server_address)
    sock.connect(server_address)

    try:
        # Send data
        print('sending img')
        # img = np.transpose(img, (2, 0 , 1))
        # img = np.zeros((100, 100, 3), dtype=np.uint8)
        print(img.shape)
        head = [0xAA, 0x55]
        size = [img.shape[1], img.shape[0]]
        sock.sendall(bytearray(head))
        sock.sendall(bytearray(size))
        sock.sendall(img.tobytes())
    finally:
        print('closing socket')
        sock.close()

sio = socketio.Client()

@sio.on('connect')
def connect():
    print('connection established')

@sio.on('disconnect')
def disconnect():
    print('disconnected from server')

@sio.on('patch')
def patch(data):
    print('Received Patch')
    image = Image.open(BytesIO(base64.b64decode(data['data'])))
    image_np = np.array(image)
    box = data['boxes'][0]
    patch = image_np[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]
    patch = Image.fromarray(patch)
    patch.show()
    send_img('192.168.199.142', 9191, np.array(patch).astype(np.uint8))

sio.connect('http://192.168.199.100:9090')

image = Image.open('camera/lena.bmp')
print(np.array(image).shape)
send_img('192.168.199.142', 9191, np.array(image).astype(np.uint8))
