# Created by Sinclert Pérez & Silvia Barbero


from cv2 import CascadeClassifier

from image_trans import greyscale_array
from image_trans import detect_face
from image_trans import cut_face
from image_trans import normalize_colors
from image_trans import resize


FACE_DETECTOR = CascadeClassifier('../resources/face_models/frontal_face.xml')




def check_face(image, face_detector = FACE_DETECTOR):

	""" Checks if an image has a face and cuts it if it does

	Arguments:
	----------
		image:
			type: PIL image
			info: image to search for a face

		face_detector:
			type: CascadeClassifier object
			info: face detector classifier

	Returns:
	----------
		image:
			type: PIL image
			info: cut image in greyscale if there is a face, None otherwise
	"""

	image = greyscale_array(image, 'BGR') # TODO: CHECK
	results = detect_face(image, face_detector)

	if results['found']:
		return cut_face(image, results)
	else:
		return None




def normalize_face(image):

	""" Normalize the greyscale distribution and the size of a given image

	Arguments:
	----------
		image:
			type: PIL image
			info: greyscale face image

	Returns:
	----------
		image:
			type: PIL image
			info: normalized greyscale and size image
	"""

	image = normalize_colors(image)
	image = resize(image)

	return image
