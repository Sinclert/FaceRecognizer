# Created by Sinclert Pérez & Silvia Barbero


import cv2

from image_process import check_faces
from image_process import draw_rect
from image_process import draw_text
from image_process import normalize_face

from utils import compute_path




def identify_actors(video_path, clf, clf_th, out_name):

	""" Identifies the actors using a classifier and generates an output video

	Arguments:
	----------
		video_path:
			type: string
			info: path to where the video is stored

		clf:
			type: FaceRecognizer object
			info: trained classifier to identify faces

		clf_th:
			type: int / float
			info: threshold to identify a face as 'Unknown'

		out_name:
			type: string
			info: name of the generated video file
	"""

	video = cv2.VideoCapture(video_path)

	# The new generated video is prepared to be saved
	frame_w = int(video.get(3))
	frame_h = int(video.get(4))

	out_path = compute_path(out_name + '.avi', 'video')
	out = cv2.VideoWriter(
		filename = out_path,
		fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
		fps = 20,
		frameSize = (frame_w, frame_h)
	)


	notFinished, frame = video.read()
	while notFinished:

		# Modify the frame for each detected face
		frame = modify_frame(frame, clf, clf_th)

		# Write the frame into the new video file
		out.write(frame)
		notFinished, frame = video.read()


	video.release()
	out.release()




def modify_frame(frame, clf, clf_th):

	""" Plot a rectangle and a label if a face is identified in the frame

	Arguments:
	----------
		frame:
			type: numpy.array
			info: array of frame pixels

		clf:
			type: FaceRecognizer object
			info: trained classifier to identify faces

		clf_th:
			type: int / float
			info: threshold to identify a face as 'Unknown'

	Returns:
	----------
		frame:
			type: numpy.array
			info: array of frame pixels (may be modified)
	"""

	for face, coords in check_faces(frame):
		face = normalize_face(face)
		label = clf.predict(face, clf_th)
		frame = draw_rect(frame, coords)
		frame = draw_text(frame, label, coords)

	return frame
