# Created by Sinclert Pérez & Silvia Barbero


import cv2
import numpy




def greyscale_array(image, color_mode):

	""" Transforms a PIL Image to a greyscale numpy array

	Arguments:
	----------
		image:
			type: PIL Image
			info: image to be transformed

		color_mode:
			type: string
			info: image color format {'RGB', 'BGR'}

	Returns:
	----------
		grey_image:
			type: numpy array
			info: grey scale numpy array
	"""

	image = numpy.array(image)

	if color_mode == 'RGB':
		grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		return grey_image

	elif color_mode == 'BGR':
		grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return grey_image

	else:
		exit('Invalid color format')




def detect_face(image, classifier, **parameters):

	""" Detects the position of a face in the given image

	Arguments:
	----------
		image:
			type: numpy array
			info: image where the face should be detected

		classifier:
			type: OpenCV CascadeClassifier
			info: classifier used to detect the face

		parameters:
			type: keyword arguments
			info: parameters of the face detection

	Returns:
	----------
		results:
			type: dict
			info: contains the following keys:
				- Found: boolean
				- X_coord:  int     (optional)
				- Y_coord:  int     (optional)
				- width:    int     (optional)
				- height:   int     (optional)
	"""

	faces = classifier.detectMultiScale(image, **parameters)

	# Only consider the first face (it must be only 1)
	if len(faces) > 0:
		(x, y, w, h) = faces[0]
		results = {
			'Found': True,
			'X_coord': x,
			'Y_coord': y,
			'width': w,
			'height': h
		}

	else:
		results = {'Found': False}

	return results




def cut_face(image, face_properties):

	""" Search for the query in a given domain and returns the html object

	Arguments:
	----------
		image:
			type: numpy array
			info: image where the face should be cut

		face_properties:
			type: dict
			info: X, Y, width and height of the detected face

	Returns:
	----------
		cut_image:
			type: numpy array
			info: image containing just the detected face
	"""

	x = face_properties['X_coord']
	y = face_properties['Y_coord']
	w = face_properties['width']
	h = face_properties['height']

	# TODO: What is this?
	w_rm = int(0.2 * w / 2)

	return image[y : (y+h) , (x+w_rm) : (x+w-w_rm)]




def normalize_colors(image):

	""" Normalize the histogram of grey values to use all of them

	Arguments:
	----------
		image:
			type: numpy array
			info: image to normalize its color values

	Returns:
	----------
		normalized_image:
			type: numpy array
			info: image with normalized colors
	"""

	return cv2.equalizeHist(image)




def resize(image, size = (100, 100)):

	""" Resize a given image to the given pixels size

	Arguments:
	----------
		image:
			type: numpy array
			info: image to resize

		size:
			type: tuple (optional)
			info: X and Y number of pixels

	Returns:
	----------
		image:
			type: numpy array
			info: new sized image
	"""

	if image.shape < size:
		interpolation = cv2.INTER_AREA
	else:
		interpolation = cv2.INTER_CUBIC

	sized_image = cv2.resize(image, size, interpolation)
	return sized_image
