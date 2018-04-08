# Created by Sinclert Pérez & Silvia Barbero


import cv2

from utils import get_file_paths


ALGORITHMS = {
	'Eigen': cv2.face.EigenFaceRecognizer_create(),
	'Fisher': cv2.face.FisherFaceRecognizer_create(),
	'LBPH': cv2.face.LBPHFaceRecognizer_create(),
}




def prepare_feats(dataset_folder):

	""" Builds a list of images arrays in order to be used as features

	Arguments:
	----------
		dataset_folder:
			type: string
			info: name of the folder with the images

	Returns:
	----------
		feats:
			type: list
			info: contains the image numpy arrays as features
	"""

	images_paths = get_file_paths(
		folder_name = dataset_folder,
		file_type = 'dataset'
	)

	feats = [cv2.imread(path, 0) for path in images_paths]
	return feats




def train(algorithm, feats, labels):

	"""

	Arguments:
	----------
		algorithm:
			type:
			info:

		feats:
			type:
			info:

		labels:
			type:
			info:

	Returns:
	----------
		model:
			type:
			info:
	"""

	model = ALGORITHMS[algorithm]
	model.train(feats, labels)

	return model
