# Created by Sinclert Pérez & Silvia Barbero


import json
import os


project_paths = {
	'dataset': ['resources', 'datasets'],
	'model': ['models'],
	'scraping_c': ['configs', 'scraping'],
	'training_c': ['configs', 'training'],
	'face_model': ['resources', 'face_models']
}




def check_keys(keys, data_struct, error):

	""" Checks if all the keys are present in the data structure

	Arguments:
	----------
		keys:
			type: list
			info: elements which must be in the data structure

		data_struct:
			type: set / dictionary
			info: data structure to check existence

		error:
			type: string
			info: error message to print
	"""

	if not all(k in data_struct for k in keys):
		exit(error)




def compute_path(file_name, file_type):

	""" Builds the absolute path to the desired file given its file type

	Arguments:
	----------
		file_name:
			type: string
			info: desired file name

		file_type:
			type: string
			info: {'dataset', 'model', 'profile_p', 'profile_t', 'stopwords'}

	Returns:
	----------
		path:
			type: string
			info: absolute path to the desired file
	"""

	try:
		project_root = str(os.path.dirname(os.getcwd()))

		path = [project_root]
		path = path + project_paths[file_type]
		path = path + [file_name]

		return os.path.join(*path)

	except KeyError:
		exit('The file type "' + file_type + '" is not defined')




def get_file_paths(folder_name, file_type):

	""" Obtains the list of file paths for a given folder

	Arguments:
	----------
		folder_name:
			type: string
			info: name of the folder whose file will be returned

		file_type:
			type: string
			info: type of files in order to know the relative path

	Returns:
	----------
		file_paths:
			type: list
			info: contains the path to each file in a given folder
	"""

	folder_path = compute_path(folder_name, file_type)

	file_names = os.listdir(path = folder_path)
	file_paths = [os.path.join(folder_path, file) for file in file_names]

	return file_paths




def load_clf(clf, file_name, file_type):

	""" Loads a classifier object with the specified name

	Arguments:
	----------
		clf:
			type: FaceRecognizer object
			info: empty object to use the 'load' function

		file_name:
			type: string
			info: saved classifier object name

		file_type:
			type: string
			info: used to determine the proper path

	Returns:
	----------
		clf:
			type: OpenCV Recognizer
			info: trained classifier model object
	"""

	file_path = compute_path(file_name, file_type)

	try:
		return clf.model.load(file_path)

	except IOError:
		exit('The object could not be loaded from ' + file_path)




def save_clf(clf, file_name, file_type):

	""" Saves a classifier object in the specified path

	Arguments:
	----------
		clf:
			type: OpenCV Recognizer
			info: trained classifier model object

		file_name:
			type: string
			info: saved classifier object name

		file_type:
			type: string
			info: used to determine the proper path
	"""

	file_path = compute_path(file_name, file_type)

	file_dir = file_path.replace(file_name, '')
	os.makedirs(file_dir, exist_ok = True)

	try:
		clf.save(file_path)
	except IOError:
		exit('The object could not be saved in ' + file_path)




def read_json(file_name, file_type):

	""" Reads a JSON file and returns it as a dictionary

	Arguments:
	----------
		file_name:
			type: string
			info: readable file name

		file_type:
			type: string
			info: used to determine the proper path

	Returns:
	----------
		json_dict:
			type: dict
			info: dictionary containing the parsed JSON file
	"""

	file_path = compute_path(file_name, file_type)

	try:
		file = open(file_path, 'r', encoding = 'utf-8')
		json_dict = json.load(file)
		file.close()

		return json_dict

	except IOError:
		exit('The file ' + file_name + ' cannot be opened')




def write_json(dictionary, file_name, file_type):

	""" Writes a dictionary object as a JSON file

	Arguments:
	----------
		dictionary:
			type: dict
			info: object to store as a JSON file

		file_name:
			type: string
			info: saved JSON name

		file_type:
			type: string
			info: used to determine the proper path
	"""

	file_path = compute_path(file_name, file_type)

	file_dir = file_path.replace(file_name, '')
	os.makedirs(file_dir, exist_ok = True)

	try:
		file = open(file_path, 'w', encoding = 'utf-8')
		json.dump(dictionary, file, sort_keys = True, indent = 4)
		file.close()

	except IOError:
		exit('The file ' + file_name + ' cannot be opened')
