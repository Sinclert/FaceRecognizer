# Created by Sinclert Pérez & Silvia Barbero


import cv2

from image_trans import greyscale_array
from image_trans import detect_face
from image_trans import cut_face
from image_trans import normalize_colors
from image_trans import resize


FACE_DETECTOR = cv2.CascadeClassifier('../resources/face_models/frontal_face.xml')




def check_faces(image, face_detector = FACE_DETECTOR):

	""" Checks if an image has a face and cuts it if it does

	Arguments:
	----------
		image:
			type: PIL image
			info: image to search for a face

		face_detector:
			type: CascadeClassifier object
			info: face detector classifier

	Yields:
	----------
		face:
			type: numpy array
			info: cut face image in greyscale

		results:
			type: dict
			info: contains the following keys:
				- X_coord:  int
				- Y_coord:  int
				- width:    int
				- height:   int
	"""

	image = greyscale_array(image, 'BGR') # TODO: CHECK
	results = detect_face(image, face_detector, scaleFactor = 1.3, minNeighbors = 4)

	for coords in results:
		face = cut_face(image, coords)
		yield face, coords




def draw_rect(image, coordinates):

	""" Draws a rectangle in a given image

	Arguments:
	----------
		image:
			type: numpy array
			info: RGB colored image

		coordinates:
			type: dict
			info: contains the following keys:
				- X_coord:  int
				- Y_coord:  int
				- width:    int
				- height:   int

	Returns:
	----------
		image:
			type: numpy array
			info: RGB colored image with a rectangle
	"""

	x = coordinates['X_coord']
	y = coordinates['Y_coord']
	w = coordinates['width']
	h = coordinates['height']

	cv2.rectangle(
		img = image,
		pt1 = (x, y),
		pt2 = (x+w, y+h),
		color = (255, 0, 0),
		thickness = 2
	)

	return image




def draw_text(image, text, coordinates):

	""" Writes some text in the given image

	Arguments:
	----------
		image:
			type: numpy array
			info: RGB colored image

		text:
			type: string
			info: text to write in the image

		coordinates:
			type: dict
			info: contains the following keys:
				- X_coord:  int
				- Y_coord:  int
				- width:    int
				- height:   int

	Returns:
	----------
		image:
			type:
			info:
	"""

	cv2.putText(
		img = image,
		text = text,
		org = (coordinates['X_coord'], coordinates['Y_coord']),
		fontFace = cv2.FONT_HERSHEY_PLAIN,
		fontScale = 3,
		color = (255, 0, 0),
		thickness = 2
	)

	return image




def normalize_face(image):

	""" Normalize the greyscale distribution and the size of a given image

	Arguments:
	----------
		image:
			type: numpy array
			info: greyscale face image

	Returns:
	----------
		image:
			type: numpy array
			info: normalized greyscale and sized image
	"""

	image = normalize_colors(image)
	image = resize(image)

	return image
