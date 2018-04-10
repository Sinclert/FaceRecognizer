# Created by Sinclert Pérez & Silvia Barbero


import cv2

from image_process import check_face
from image_process import draw_rect
from image_process import draw_text
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

	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(video.get(3))
	frame_height = int(video.get(4))

	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('outpy.avi',
	                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
	                      (frame_width, frame_height))

	while notFinished:

		face, coords = check_face(frame)

		# If there is a face: predicts the face label
		if face is not None:
			face = normalize_face(face)
			label = clf.predict(face, clf_th)
			frame = draw_rect(frame, coords)
			frame = draw_text(frame, label, coords)

		# Write the frame into the file 'output.avi'
		out.write(frame)
		notFinished, frame = video.read()

	video.release()
	out.release()




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
