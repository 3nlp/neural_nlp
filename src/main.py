"""
Main Functionalities
@
@
@
"""

import numpy as np
from controllers import data_extractor

def main():

	test_file_name = 'data/test.json'
	transformed_data = data_extractor.import_in_order(test_file_name)

if __name__ == "__main__":
    main()


