# Created by Sinclert Pérez & Silvia Barbero


import cv2

from image_process import check_face
from image_process import normalize_face




def identify_actors(video_path, clf, clf_th):

	"""

	Arguments:
	----------
		video_path:
			type:
			info:

		clf:
			type:
			info:

		clf_th:
			type:
			info:
	"""

	video = cv2.VideoCapture(video_path)
	notFinished, frame = video.read()

	while notFinished:

		face = check_face(frame)

		# If no face is detected: continue
		if face is None: continue

		# Normalizes and predicts the face label
		face = normalize_face(face)
		label, conf = clf.predict(face)
		print(label)

		notFinished, frame = video.read()

	video.release()




def save_video(video, output):

	"""

	Arguments:
	----------
		video:
			type:
			info:

		output:
			type:
			info:
	"""

	pass
