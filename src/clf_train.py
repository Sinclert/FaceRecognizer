# Created by Sinclert Pérez & Silvia Barbero


import cv2


ALGORITHMS = {
	'Eigen': cv2.face.EigenFaceRecognizer_create(),
	'Fisher': cv2.face.FisherFaceRecognizer_create(),
	'LBPH': cv2.face.LBPHFaceRecognizer_create(),
}




def prepare_feats(dataset_folder):
	pass


def train(algorithm, feats, labels):
	pass
