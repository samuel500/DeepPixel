from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from flask_wtf.file import FileField, FileRequired

from wtforms.validators import DataRequired, Length, EqualTo, ValidationError


# class ContentForm(FlaskForm):
#     title = StringField('Title', validators=[DataRequired()])
#     link = StringField('Link', validators=[DataRequired()])
#     description = TextAreaField('Content', validators=[])
#     add = SubmitField('Add')


class DreamForm(FlaskForm):

	#photo = FileField(validators=[FileRequired()])
    title = StringField('Title', validators=[DataRequired(), Length(min=2, max=30)])

	#add = SubmitField('Dreamify')