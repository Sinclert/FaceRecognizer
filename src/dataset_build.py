# Created by Sinclert Pérez & Silvia Barbero


from dataset_miner import build_url
from dataset_miner import get_page
from dataset_miner import get_next_page
from dataset_miner import get_images

from image_process import check_face
from image_process import normalize_face


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

		for img in get_images(page):
			face = check_face(img)

			# If no face is detected: continue
			if face is None: continue

			# Normalizes and saves the image
			face = normalize_face(face)
			save_image(face)
			stored += 1




def save_image(image):

	# TODO

	pass
