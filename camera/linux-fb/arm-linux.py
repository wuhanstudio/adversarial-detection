import argparse
import socketio
from time import sleep

# Data IO and Encoding-Decoding
from io import BytesIO
import base64

# Image Processing
from PIL import Image

import numpy as np

sio = socketio.Client()

# From image to base64 string
def img2base64(image):
    origin_img = Image.fromarray(np.uint8(image))
    origin_buff = BytesIO()
    origin_img.save(origin_buff, format="JPEG")

    return base64.b64encode(origin_buff.getvalue()).decode("utf-8")

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
    img_fb = np.array(Image.new('RGBA', size=(1920, 1080)))
    img_fb[int(1080/2-416/2):int(1080/2+416/2), int(1920/2-416/2):int(1920/2+416/2), :3] = image_np[:, :, :]

    fb = np.memmap('/dev/fb0', dtype='uint8',mode='w+', shape=(1080, 1920, 4))
    fb[:, :, :] = img_fb[:, :, :]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Raspi Framebuffer Client')
    parser.add_argument('--ip', help='Server IP Address', type=str, required=True)
    parser.add_argument('--port', help='Server Port', type=str, required=True)
    args = parser.parse_args()

    sio.connect('http://' + args.ip + ':' + args.port)
