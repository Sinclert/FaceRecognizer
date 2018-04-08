# Created by Sinclert Pérez & Silvia Barbero


import cv2
import numpy

from utils import get_file_paths
from utils import load_object


ALGORITHMS = {
	'Eigen': cv2.face.EigenFaceRecognizer_create(),
	'Fisher': cv2.face.FisherFaceRecognizer_create(),
	'LBPH': cv2.face.LBPHFaceRecognizer_create(),
}




class FaceClassifier(object):


	""" Represents a face classifier

	Attributes:
	----------
		model:
			type: OpenCV Recognizer
			info: trained classifier model

		labels_dict:
			type: dict
			info: contains the relation integer <-> actor name
	"""




	def __init__(self, file_name = None):

		""" Loads a trained model or initiates a new one

		Arguments:
		----------
			file_name:
				type: string (optional)
				info: name of the saved model file
		"""

		if file_name is not None:
			self.__dict__ = load_object(file_name, 'model')

		else:
			self.model = None
			self.labels_dict = {}




	def __prepare_feats(self, datasets_info):

		""" Builds the feature and label vectors from the specified datasets

		Arguments:
		----------
			datasets_info:
				type: list
				info: list of dictionaries containing:
					- label (string)
					- folder (string)

		Returns:
		----------
			feats:
				type: numpy.array
				info: contains all the image arrays

			labels:
				type: numpy.array
				info: contains all the integer-encoded labels
		"""

		feats, labels = [], []

		for i, dataset in enumerate(datasets_info):

			images_paths = get_file_paths(
				folder_name = dataset['folder'],
				file_type = 'dataset'
			)

			feats += [cv2.imread(path, 0) for path in images_paths]
			labels += [i] * len(images_paths)

			# Updates the corresponding ints <-> labels dict
			self.labels_dict[i] = dataset['label']


		return feats, numpy.array(labels)




	def predict(self, frame):

		"""

		Arguments:
		----------
			frame:
				type:
				info:

		Returns:
		----------
			label:
				type:
				info:
		"""

		# TODO




	def train(self, algorithm, datasets_info):

		""" Trains the specified OpenCV Recognizer algorithm

		Arguments:
		----------
			algorithm:
				type: string
				info: name of the algorithm {Eigen, Fisher, LBPH}

			datasets_info:
				type: list
				info: dictionaries containing datasets labels and folder
		"""

		feats, labels = self.__prepare_feats(datasets_info)

		try:
			self.model = ALGORITHMS[algorithm]
			self.model.train(feats, labels)

		except KeyError:
			exit('Invalid algorithm name')
