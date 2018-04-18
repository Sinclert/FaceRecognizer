# Created by Sinclert Pérez & Silvia Barbero


import cv2
import numpy
import random

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
			self.properties = {
				'algorithm': algorithm,
				'labels': {}
			}

		except KeyError:
			exit('Invalid algorithm name')




	def __prepare_samples(self, datasets_info):

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
			self.properties['labels'][i] = dataset['label']


		return feats, numpy.array(labels)




	def __validate(self, samples, labels, test_pct = 0.2):

		""" Validates the trained model using a train-test samples separation

		Arguments:
		----------
			samples:
				type: list
				info: contains all the images

			labels:
				type: list
				info: contains all the images labels

			test_pct:
				type: float (optional)
				info: percentage of samples used to test
		"""

		if test_pct < 0 or test_pct > 1:
			exit('Invalid test percentage to validate')


		# Shuffle the samples and the labels
		comb_list = list(zip(samples, labels))
		random.shuffle(comb_list)
		samples, labels = zip(*comb_list)

		# Split by the desired test percentage
		split = int(len(samples) * test_pct)

		train_samples = samples[split:]
		train_labels = numpy.array(labels[split:])
		test_samples = samples[:split]
		test_labels = numpy.array(labels[:split])


		# Training partial model
		print('Starting validation process')
		validation_model = ALGORITHMS[self.properties['algorithm']]
		validation_model.train(train_samples, train_labels)

		total, good = len(test_samples), 0

		# Testing over the test_samples
		for i, sample in enumerate(test_samples):
			label, _ = self.model.predict(sample)

			# If the predicted label is the correct one
			if label == test_labels[i]: good += 1

		print('Accuracy:', round(good/total, 2))




	def predict(self, image, clf_th):

		""" Predicts a label (name) for a given face image

		Arguments:
		----------
			frame:
				type: numpy.array
				info: normalized greyscale and sized image

			clf_th:
				type: float
				info: confidence percentage threshold

		Returns:
		----------
			label:
				type: string
				info: name of the actor
		"""

		label, conf = self.model.predict(image)

		if conf >= clf_th:
			label = self.properties['labels'][str(label)]
		else:
			label = 'Unknown'

		print(label, round(conf, 4))
		return label




	def train(self, datasets_info, validate = True):

		""" Trains the specified OpenCV Recognizer algorithm

		Arguments:
		----------
			datasets_info:
				type: list
				info: dictionaries containing datasets labels and folder

			validate
				type: bool
				info: indicates if the model should be validated
		"""

		samples, labels = self.__prepare_samples(datasets_info)
		self.model.train(samples, labels)

		# Validation process
		if validate: self.__validate(
			samples = samples,
			labels = labels,
		)
