from utils import txt_2_json, create_input_files
from config import *
import os.path

if __name__ == '__main__':
    # convert custom txt caption to karpathy json format, only once
    if not os.path.isfile(caption_json_path):
        txt_2_json()

    # Create input files (along with word map)
    create_input_files()
