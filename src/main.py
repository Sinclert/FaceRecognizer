# Created by Sinclert Pérez & Silvia Barbero


from argparse import ArgumentParser as Parser
from argparse import RawDescriptionHelpFormatter

from clf_train import prepare_feats
from clf_train import train

from dataset_build import create_dataset
from utils import read_json
from utils import save_object


# Default CLI modes
modes = [
	'analyse_video',
	'build_datasets',
	'train_model'
]




def analyse_video(video_path, model_name, clf_th, frames_th):

	"""

	Arguments:
	----------
	"""

	# TODO




def build_datasets(dataset_config):

	""" Builds datasets of pictures given a configuration file

	Arguments:
	----------
		dataset_config:
			type: string
			info: name of the JSON with the dataset configuration
	"""

	datasets = read_json(
		file_name = dataset_config,
		file_type = 'scraping_c'
	)

	for data in datasets:
		create_dataset(data['actor_query'], data['pics_number'])




def train_model(algorithm, training_config, output):

	""" Trains a OpenCV classifier and stores it in the models folder

	Arguments:
	----------
		algorithm:
			type: string
			info: name of the classifier {Eigen, Fisher, LBPH}

		training_config:
			type: string
			info: name of the JSON with the training configuration

		output:
			type: string
			info: name of the output model
	"""

	datasets = read_json(
		file_name = training_config,
		file_type = 'training_c'
	)

	feats, labels = [], []

	for dataset_folder, label in datasets.items():
		new_feats = prepare_feats(dataset_folder)
		feats += new_feats
		labels += [label] * len(new_feats)

	model = train(
		algorithm = algorithm,
		feats = feats,
		labels = labels
	)

	save_object(model, output, 'model')




if __name__ == '__main__':

	global_parser = Parser(
		usage = 'main.py [mode] [arguments]',
		description =
			'modes and arguments:\n'
			'  \n'
			'  analyse_video: generates a copy of a video with the actors faces\n'
			'			-v <video path>\n'
			'			-m <model name>\n'
			'			-c <classifier threshold>\n'
			'			-f <frames threshold>\n'
			'  \n'
			'  build_datasets: builds several datasets of actors faces\n'
			'			-d <datasets config file>\n'
			'  \n'
			'  train_model: trains a OpenCV model\n'
			'			-a <algorithm name>\n'
			'			-d <training config file>\n'
			'			-o <output name>\n',
		formatter_class = RawDescriptionHelpFormatter
	)

	# Parsing the arguments in order to check the mode
	global_parser.add_argument('mode', choices = modes)
	arg, func_args = global_parser.parse_known_args()


	if arg.mode == 'analyse_video':

		parser = Parser(usage = "Use 'main.py -h' for help")
		parser.add_argument('-v', required = True)
		parser.add_argument('-m', required = True)
		parser.add_argument('-c', required = True, type = float)
		parser.add_argument('-f', required = True, type = int)

		args = parser.parse_args(func_args)
		analyse_video(args.v, args.m, args.c, args.f)


	elif arg.mode == 'build_datasets':

		parser = Parser(usage = "Use 'main.py -h' for help")
		parser.add_argument('-d', required = True)

		args = parser.parse_args(func_args)
		build_datasets(args.d)


	elif arg.mode == 'train_model':

		parser = Parser(usage = "Use 'main.py -h' for help")
		parser.add_argument('-a', required = True)
		parser.add_argument('-d', required = True)
		parser.add_argument('-o', required = True)

		args = parser.parse_args(func_args)
		train_model(args.a, args.d, args.o)
