# Created by Sinclert Pérez & Silvia Barbero


import cv2
import numpy

from utils import get_file_paths


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




	def __init__(self, algorithm):

		""" Initiates a classifier object with a model type

		Arguments:
		----------
			algorithm:
				type: string
				info: name of the algorithm {Eigen, Fisher, LBPH}
		"""

		try:
			self.model = ALGORITHMS[algorithm]
			self.labels_dict = {}

		except KeyError:
			exit('Invalid algorithm name')




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




	def train(self, datasets_info):

		""" Trains the specified OpenCV Recognizer algorithm

		Arguments:
		----------
			datasets_info:
				type: list
				info: dictionaries containing datasets labels and folder
		"""

		feats, labels = self.__prepare_feats(datasets_info)

		self.model.train(feats, labels)