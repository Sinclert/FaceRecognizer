# Created by Sinclert Pérez & Silvia Barbero


import os

from PIL import Image

from dataset_miner import build_url
from dataset_miner import get_page
from dataset_miner import get_next_page
from dataset_miner import get_images

from image_process import check_face
from image_process import normalize_face

from utils import compute_path


SEARCH_ENGINE = {
	'domain': 'www.google.com',
    'path': '/search',
    'params': {
		'tbm': 'isch'
	}
}




def create_dataset(query, pics_num, search_engine = SEARCH_ENGINE):

	""" Downloads, transforms and stores pictures given a query and an engine

	Arguments:
	----------
		query:
			type: string
			info: query that fill be plot into the search engine

		num:
			type: int
			info: number of pictures to obtain

		search_engine:
			type: dict (optional)
			info: properties of the search engine to use. Keys:
				- domain (string)
				- path (string)
				- params (dict)
	"""

	# Unwrapping search engine properties
	domain = search_engine['domain']
	path = search_engine['path']
	params = search_engine['params']
	params['q'] = query

	stored = 0

	# Until the number of pictures is not reached
	while stored < pics_num:

		url = build_url(domain, path, params)
		page = get_page(url)

		path = get_next_page(page, 'fl')
		params = None

		# Each image is saved if a face is detected
		for image in get_images(page):
			face, _ = check_face(image)

			# If no face is detected: continue
			if face is None: continue

			# Normalizes and stores the image
			face = Image.fromarray(normalize_face(face))
			save_image(
				image = face,
				output_folder = query.replace(' ', '_'),
				output_name = str(stored)
			)

			stored += 1
			if stored == pics_num: break




def save_image(image, output_folder, output_name):

	""" Stores the given image in the output path with the output name

	Arguments:
	----------
		image:
			type: PIL image
			info: image to store

		output_path:
			type: string
			info: folder to store the image

		output_name:
			type: string
			info: name of the image file
	"""

	folder_path = compute_path(output_folder, 'dataset')
	os.makedirs(folder_path, exist_ok = True)

	file_path = os.path.join(folder_path, output_name + '.png')
	image.save(file_path)
