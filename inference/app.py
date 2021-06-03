from flask import Flask
from flask_restful import Api, Resource, abort, reqparse
from flasgger import Swagger, swag_from
from fastai.learner import load_learner
from fastai.vision import *
from fastai.vision.core import *
from werkzeug.datastructures import FileStorage
from io import BytesIO



app = Flask(__name__)
api = Api(app)
app.config['SWAGGER'] = {
    'title': 'Covid XRay Model',
    'uiversion': 3
}
swag = Swagger(app)

parser = reqparse.RequestParser()
parser.add_argument('covidxray')

def label_func(x): return x.parent.name # get the folder the file is in for a label

class COVIDXRay(Resource):

  def __init__(self):
    self.parser = reqparse.RequestParser()
    self.model = load_learner('./models/multi-class-pg.pkl')

  def label_func(x): return x.parent.name # get the folder the file is in for a label

  def post(self):
      """
      Validate data meets requirements
      ---
      tags:
        - COVIDXRay
      consumes: [multipart/form-data]
      parameters:
          - name: covidxray
            in: formData
            required: true
            type: file
            description: An x-ray image
          - name: label
            in: formData
            required: true
            type: string
            description: The real label for testing
      responses:
        201:
          description: The image was processed
      """

      self.parser.add_argument('covidxray', type=FileStorage, location='files', required=True)
      self.parser.add_argument('label', type=str, required=True)

      args = self.parser.parse_args()
      uploadedImage = load_image(BytesIO(args['covidxray'].read())).reshape(256,256) # this may not work for all images
      #img = load_image(uploadedImage)

      pred_class,pred_idx,outputs = self.model.predict(PILImage(uploadedImage))
      i = pred_idx.item()
      classes = ['covid', 'nofinding', 'pneumonia']
      prediction = classes[i]
      result = {'prediction':prediction,
                'label':args['label']}

      return result, 201

api.add_resource(COVIDXRay, '/covidxray')

# Threaded false. Debug false. Trying to get things working.
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False)