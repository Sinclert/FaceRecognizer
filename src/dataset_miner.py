# Created by Sinclert Pérez & Silvia Barbero


import base64
import io
import requests

from bs4 import BeautifulSoup
from PIL import Image




def build_url(domain, path, parameters = None):

	""" Search for the query in a given domain and returns the html object

	Arguments:
	----------
		domain:
			type: string
			info: search engine to which the query is requested

		path:
			type: string
			info: relative path within the domain

		parameters:
			type: dict
			info: any additional keyword arguments required by the search engine

	Returns:
	----------
		url:
			type: string
			info: link to a specific web page
	"""

	url = 'https://' + domain + path

	if parameters is None:
		url += '?'

		for key, value in parameters.items():
			url += key + '=' + value.replace(' ', '+') + '&'

	return url




def get_page(url):

	""" Request a specific url content and returns its html

	Arguments:
	----------
		url:
			type: string
			info: link to a specific web page

	Returns:
	----------
		page:
			type: BeautifulSoup object
			info: html of the page in soup format
	"""

	response = requests.get(url)
	page = BeautifulSoup(response.text, 'lxml')

	return page




def get_next_page(page, class_name):

	""" Obtain the url from the html page to jump into the next results page

	Arguments:
	----------
		page:
			type: string (html format)
			info: html of the page containing the images

		class_name:
			type: string
			info: name of the div containing the link to the next page

	Returns:
	----------
		url:
			type: string
			info: link to the next results page
	"""

	urls = page.findAll('a', {'class': class_name})

	return urls[-1]['href']




def get_images_src(page):

	""" Yields each image source link from the html page (Generator)

	Arguments:
	----------
		page:
			type: string (html format)
			info: html of the page containing the images

		class_name:
			type: string
			info: name of the pictures class (defined by search engine)

	Yields:
	----------
		source:
			type: string
			info: source link of an image
	"""

	images = page.findAll('img')

	for img in images:
		print(img['src'])




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
