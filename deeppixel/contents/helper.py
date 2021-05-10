import secrets
from PIL import Image, ExifTags
import urllib.request, urllib.error, urllib.parse
from flask import url_for, current_app
import json
import base64
import os


def content_interactions_html_dict(interaction):
    if interaction:
        buttons_list = {'star': interaction.star}
    else:
        buttons_list = {'star': False}
    for k in buttons_list:
        if buttons_list[k]:
            buttons_list[k] = "content-library-button-selected"
        else:
            buttons_list[k] = "content-library-button-unselected"
    return buttons_list



def save_picture(form_picture, m_dim=800):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(current_app.root_path, 'static/site_pics', picture_fn)



    # output_size = (125, 125)
    i = Image.open(form_picture)
    if hasattr(i, '_getexif'):
        exif = i._getexif()
        if exif:
            for tag, label in ExifTags.TAGS.items():
                if label == 'Orientation':
                    orientation = tag
                    break
            if orientation in exif:
                if exif[orientation] == 3:
                    i = i.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    i = i.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    i = i.rotate(90, expand=True)

    m = max(i.size)
    if m > m_dim:
        r = 800/m_dim
        i = i.resize((int(i.size[0]*r), int(i.size[1]*r)), Image.BICUBIC)
    # i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn
