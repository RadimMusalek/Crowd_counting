'''
Configs
'''

from pathlib import Path


class Config(object):

    # Directory paths
    CURRENT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = CURRENT_DIR.parent
    DATA_DIR = ROOT_DIR.joinpath("data")
    LOGS_DIR = ROOT_DIR.joinpath("logs")
    MODELS_DIR = ROOT_DIR.joinpath("models")
    PROJECT_SCRIPT_FILES_DIR = ROOT_DIR.joinpath("project_script_files")
    RESULTS_DIR = ROOT_DIR.joinpath("results")

    #  ADD OTHER CONFIGS


config = Config()
