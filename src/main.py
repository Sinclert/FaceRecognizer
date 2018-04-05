# Created by Sinclert Pérez & Silvia Barbero


from argparse import ArgumentParser as Parser
from argparse import RawDescriptionHelpFormatter


# Default CLI modes
modes = [
	'analyse_video',
	'build_dataset',
	'train_model'
]




def analyse_video(video_path, model_name, clf_th, frames_th):

	"""

	Arguments:
	----------
	"""

	# TODO




def build_datasets(dataset_config):

	"""

	Arguments:
	----------
	"""

	# TODO




def train_model(algorithm, training_config, output):

	"""

	Arguments:
	----------
	"""

	# TODO




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
