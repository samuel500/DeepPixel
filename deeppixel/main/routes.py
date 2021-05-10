from flask import render_template, request, Blueprint
from flask_login import current_user
from deeppixel.models import Content, UserLibrary
from sqlalchemy import and_
from deeppixel.contents.helper import content_interactions_html_dict


main = Blueprint('main', __name__)

@main.route("/")
@main.route("/home")
def home():
    page = request.args.get('page', default=1, type=int)
    content_per_page = 10
    contents = Content.query.order_by(Content.date_added.desc()).filter(Content.published.is_(True)).paginate(page=page, per_page=content_per_page)
    page_obj = contents
    buttons_info = []
    user_info = []
    if current_user.is_authenticated:
        for content in contents.items:
            interactions = UserLibrary.query.filter(and_(UserLibrary.content_id==content.content_id, UserLibrary.user_id==current_user.id)).first()
            user_info.append(interactions)
        for i in range(len(user_info)):
            buttons_list = content_interactions_html_dict(user_info[i])
            buttons_info.append(buttons_list)

    else:
        for i in range(len(contents.items)):
            buttons_list = {'consuming': False, 'star': False, 'to_consume': False, 'consumed': False, 'add_to_library': False}

            for k in buttons_list:
                if buttons_list[k]:
                    buttons_list[k] = "content-library-button-selected"
                else:
                    buttons_list[k] = "content-library-button-unselected"
            buttons_info.append(buttons_list)
    contents = zip(contents.items, buttons_info)
    return render_template('home.html', contents=contents, page_obj=page_obj)


@main.route("/about")
def about():
    return render_template('about.html', title='About')