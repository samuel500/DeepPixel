from flask import (render_template, url_for, flash, redirect, request, abort, jsonify, Blueprint, send_file)
from flask_login import current_user, login_required
from deeppixel import db
from deeppixel.models import Content, UserLibrary, User
from deeppixel.contents.forms import DreamForm
from sqlalchemy import and_
from deeppixel.contents.helper import *


from PIL import Image

import time

import io

import boto3
import botocore


contents = Blueprint('contents', __name__)


cfg = botocore.config.Config(read_timeout=900)

lambda_client = boto3.client('lambda', config=cfg)



@contents.route("/content/add", methods=['GET', 'POST'])
@login_required
def add_content():
    form = DreamForm()


    return render_template('add_content.html', form=form, legend='Add Content')



@contents.route("/content/upload_dream_image", methods=['POST'])
@login_required
def upload_dream_image():

    r = request


    uploaded_img = save_picture(r.files['img1'])


    #######
    with open(os.path.join(current_app.root_path, 'static/site_pics', uploaded_img), 'rb') as image_source:
        image_bytes = image_source.read()


    opts = {'step_size': 0.03,
                'steps_per_octave': 20,
                'num_octaves': 2,
                'octave_scale': 1.2}

    payload = {'original_image_bytes': base64.b85encode(image_bytes).decode('utf-8'), 
                'type': "dream",
                'opts': opts}

    payload = json.dumps(payload)

    st = time.time()

    
    resp = lambda_client.invoke(FunctionName='dream-lambda', Payload=payload)

    tot_t = time.time()-st
    user = User.query.filter_by(id=current_user.id).first_or_404()
    user.credits_used += tot_t
    db.session.commit()

    new_img = base64.b85decode(json.load(resp['Payload'])['dream_img'])

    im = Image.open(io.BytesIO(new_img))

    ########

    # from deeppixel.dream_utils.dreamer import dreamify
    # new_img = dreamify(os.path.join(current_app.root_path, 'static/site_pics', uploaded_img))

    # im = Image.fromarray(new_img)
    #####

    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(uploaded_img)
    dream_img = random_hex + f_ext
    picture_path = os.path.join(current_app.root_path, 'static/site_pics', dream_img)

    im.save(picture_path)



    content = Content(title="Default", content_type="dream", original_image_file=uploaded_img, generated_image_file=dream_img, creator=current_user)

    db.session.add(content)
    db.session.flush()
    c_id = content.content_id
    # print("c_id", c_id)
    db.session.commit()


    return jsonify(message="success", content_id=c_id)



@contents.route("/content/get_dream_image", methods=['POST'])
@login_required
def get_dream_image():

    content_id = request.json['content_id']


    content = Content.query.get_or_404(content_id)


    filename = os.path.join(current_app.root_path, 'static/site_pics', content.generated_image_file)

    f = send_file(filename, mimetype='image/jpeg', cache_timeout=0)
    print(f)
    return f



# @contents.route("/content/get_style_image", methods=['POST'])
# @login_required
# def get_style_image():

#     content_id = request.json['content_id']


#     content = Content.query.get_or_404(content_id)


#     filename = os.path.join(current_app.root_path, 'static/site_pics', content.generated_image_file)

#     f = send_file(filename, mimetype='image/jpeg', cache_timeout=0)
#     return f






@contents.route("/content/publish_image", methods=['POST'])
@login_required
def publish_image():

    content_id = request.json['content_id']

    content = Content.query.get_or_404(content_id)

    if content.creator != current_user: # sanity check
        abort(403)

    content.published = True
    if request.json['title']:
        content.title = request.json['title']

    db.session.commit()


    return jsonify(message="redirect", target=url_for('main.home'))



@contents.route("/content/upload_style_images", methods=['POST'])
@login_required
def upload_style_images():

    r = request


    style_img = save_picture(r.files['style_input'])

    to_style_img = save_picture(r.files['to_style_input'])


    ########
    with open(os.path.join(current_app.root_path, 'static/site_pics', to_style_img), 'rb') as image_source:
        image_bytes = image_source.read()

    with open(os.path.join(current_app.root_path, 'static/site_pics', style_img), 'rb') as image_source:
        style_image_bytes = image_source.read()


    sw = 0.08 #0.015
    opts = {
        'content_weight' : 2000, #2500, 
        'content_layers' : [2],
        'style_layers' : (2, 3, 5, 6, 9), #, 10),
        'style_weights' : (sw*20000, sw*2000, sw*400, sw*40, 2*sw), #(20, 0,0,0,32), #
        'total_variation_weight' : 0.005,
        'max_iter': 18,
        'lr': 0.2 #0.25
    }

    payload = {'original_image_bytes': base64.b85encode(image_bytes).decode('utf-8'), 
                'style_image_bytes': base64.b85encode(style_image_bytes).decode('utf-8'), 
                'type': "style",
                'opts': opts}

    payload = json.dumps(payload)


    st = time.time()
    
    resp = lambda_client.invoke(FunctionName='dream-lambda', Payload=payload)

    tot_t = time.time()-st
    user = User.query.filter_by(id=current_user.id).first_or_404()
    user.credits_used += tot_t
    db.session.commit()



    new_img = base64.b85decode(json.load(resp['Payload'])['style_img'])

    im = Image.open(io.BytesIO(new_img))
    ########

    # from deeppixel.dream_utils.styler import styler

    # new_img = styler(os.path.join(current_app.root_path, 'static/site_pics', style_img), 
    #                  os.path.join(current_app.root_path, 'static/site_pics', to_style_img))

    # im = Image.fromarray(new_img)
    #####

    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(to_style_img)
    styled_img = random_hex + f_ext
    picture_path = os.path.join(current_app.root_path, 'static/site_pics', styled_img)

    im.save(picture_path)



    content = Content(title="Default", content_type="style", original_image_file=to_style_img, generated_image_file=styled_img, style_image_file=style_img, creator=current_user)

    db.session.add(content)
    db.session.flush()
    c_id = content.content_id
    db.session.commit()


    return jsonify(message="success", content_id=c_id)



@contents.route("/content/get_style_image", methods=['POST'])
@login_required
def get_style_image():

    content_id = request.json['content_id']


    content = Content.query.get_or_404(content_id)


    filename = os.path.join(current_app.root_path, 'static/site_pics', content.generated_image_file)

    f = send_file(filename, mimetype='image/jpeg', cache_timeout=0)
    return f




@contents.route("/content/user_interaction", methods=['POST'])
def content_user_interaction():
    if not current_user.is_authenticated:
        return jsonify(message="redirect", target=url_for('users.login'))


    button_id = request.form['button_id'].split('-')[0]
    button_id = "".join([x for x in button_id if not x.isdigit()])
    button_context = request.form['button_id'][len(button_id):]
    content_id = int(request.form['content_id'])

    record = UserLibrary.query.filter(and_(UserLibrary.content_id==content_id, UserLibrary.user_id==current_user.id)).first()

    if record:
        if button_id == 'add_to_library':
            db.session.delete(record)
            to_return = []
        else:
            setattr(record, button_id, not getattr(record, button_id))
            to_return = record
    else:
        new_library_content = UserLibrary(user_id=current_user.id, content_id=content_id, star=False)
        if button_id != 'add_to_library':
            setattr(new_library_content, button_id, not getattr(new_library_content, button_id))
        db.session.add(new_library_content)
        to_return = new_library_content

    to_return = content_interactions_html_dict(to_return)

    db.session.commit()
    return jsonify(message="success", interactions=to_return, button_context=button_context)



@contents.route("/content/<int:content_id>")
def content(content_id):
    content = Content.query.get_or_404(content_id)
    if not content.published:
        return redirect(url_for('main.home'), code=302) 

    if current_user.is_authenticated:
        interaction = UserLibrary.query.filter(and_(UserLibrary.content_id==content.content_id, UserLibrary.user_id==current_user.id)).first()
        content = [content, content_interactions_html_dict(interaction)]

    else:
        interaction = {"star": "content-library-button-unselected"}
        content = [content, interaction]
    return render_template('content.html', content=content)



@contents.route("/content/delete", methods=['POST'])
@login_required
def delete_content():
    if not current_user.is_authenticated:
        return jsonify(message="redirect", target=url_for('users.login'))

    content_id = int(request.form['content_id'])
    content = Content.query.get_or_404(content_id)
    if content.creator != current_user:
        abort(403)
    interactions = UserLibrary.query.filter(UserLibrary.content_id==content.content_id).all()
    for interaction in interactions:
        db.session.delete(interaction)
    db.session.delete(content)


    db.session.commit()
    flash('Your content has been deleted!', 'success')
    return jsonify(message="redirect", target=url_for('main.home'))
