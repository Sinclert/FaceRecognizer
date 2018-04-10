# Created by Sinclert Pérez & Silvia Barbero


import cv2

from image_process import check_face
from image_process import draw_rectangle
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

		face, coords = check_face(frame)

		# If there is a face: predicts the face label
		if face is not None:
			face = normalize_face(face)
			label = clf.predict(face, clf_th)
			frame = draw_rectangle(frame, coords)
			print(label)

		cv2.imshow('img', frame)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

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
