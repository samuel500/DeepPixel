from datetime import datetime
from deeppixel import db, login_manager
from flask_login import UserMixin
from uuid import uuid4


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class UserLibrary(db.Model):
    __tablename__ = 'user_library'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    content_id = db.Column(db.String(36), db.ForeignKey('content.content_id'), primary_key=True)

    star = db.Column(db.Boolean, default=False)
    content = db.relationship("Content", back_populates="users")
    user = db.relationship("User", back_populates="library_content")

    def __repr__(self):
        return f"UserLibrary item('content_id:{self.content_id}', 'user_id:{self.user_id}')"


class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    credits_used = db.Column(db.Float, nullable=False, default=0.0)

    content = db.relationship('Content', backref='creator', lazy='dynamic')



    library_content = db.relationship("UserLibrary", back_populates="user")


    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"


class Content(db.Model):
    __tablename__ = 'content'
    content_id = db.Column(db.String(36), primary_key=True, default=str(uuid4()).replace('-', ''))
    title = db.Column(db.String(100), nullable=False)
    date_added = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    content_type = db.Column(db.String(5), nullable=False) # "dream" or "style"

    original_image_file = db.Column(db.String(20), nullable=False, default='unavailable.jpg')
    generated_image_file = db.Column(db.String(20), nullable=False, default='unavailable.jpg')
    style_image_file = db.Column(db.String(20), nullable=False, default='unavailable.jpg')

    published = db.Column(db.Boolean, default=False)

    users = db.relationship("UserLibrary", back_populates="content")

    def __repr__(self):
        return f"Content('{self.title}', '{self.date_added}')"

