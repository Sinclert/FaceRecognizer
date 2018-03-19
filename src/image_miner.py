# Created by Sinclert Pérez & Silvia Barbero


import base64
import bs4
import io
import requests

from PIL import Image




def get_images_page(domain, query, **parameters):

	""" Search for the query in a given domain and returns the html object

	Arguments:
	----------
		domain:
			type: string
			info: search engine to which the query is requested

		query:
			type: string
			info: concept to ask the search engine

		parameters:
			type: keyword arguments
			info: any additional keyword arguments required by the search engine

	Returns:
	----------
		page:
			type: string (html format)
			info: html of the page containing the images
	"""

	# TODO




def get_images_urls(page):

	""" Yields each image source link from the html page (Generator)

	Arguments:
	----------
		page:
			type: string (html format)
			info: html of the page containing the images

	Yields:
	----------
		source:
			type: string
			info: source link of an image
	"""

	# TODO




def get_image(url):

	""" Requests the url image bytes and yields them as a PIL Image (Generator)

	Arguments:
	----------
		url:
			type: string
			info: link to a specific image

	Yields:
	----------
		img:
			type: PIL Image
			info: image from the given url
	"""

	image_bytes = requests.get(url).content

	image = base64.b64encode(image_bytes)
	image = base64.b64decode(image)
	image = Image.open(io.BytesIO(image))

	return image
