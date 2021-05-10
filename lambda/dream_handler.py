from dreamer import dreamify

from styler import styler

import os
import json

import base64
from PIL import Image


print('loading funcion')

def lambda_handler(payload, context):

	image = base64.b85decode(payload['original_image_bytes'])



	with open("/tmp/" + "tmp.jpg", 'wb') as f:
		f.write(image)

	opts = None
	if 'opts' in payload:
		opts = payload['opts']

	if payload['type'] == 'dream':


		img = dreamify("/tmp/" + "tmp.jpg", opts)

		im = Image.fromarray(img)
		im.save("/tmp/tmp2.jpg")
		with open("/tmp/tmp2.jpg", 'rb') as image_source:
			image_bytes = image_source.read()

		return {'dream_img': base64.b85encode(image_bytes).decode('utf-8')}


	elif payload['type'] == 'style':

		style_image =  base64.b85decode(payload['style_image_bytes'])

		with open("/tmp/" + "style.jpg", 'wb') as f:
			f.write(style_image)

		img = styler("/tmp/style.jpg", "/tmp/" + "tmp.jpg", opts)

		im = Image.fromarray(img)
		im.save("/tmp/tmp2.jpg")
		with open("/tmp/tmp2.jpg", 'rb') as image_source:
			image_bytes = image_source.read()


		return {'style_img': base64.b85encode(image_bytes).decode('utf-8')}

