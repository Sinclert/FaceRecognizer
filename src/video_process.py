# Created by Sinclert Pérez & Silvia Barbero


import cv2




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

	while video.isOpened():
		ret, frame = video.read()

	video.release()
	pass




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
