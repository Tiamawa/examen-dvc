import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder


def import_raw_data(local_raw_data_path, distant_folder_url):
    """ Here we import data from distant_folder_url to local_raw_data_path"""
    if check_existing_folder(local_raw_data_path):
        os.makedirs(local_raw_data_path)

    # Here we are downloading the files
    input_file = os.path.join(distant_folder_url, "raw.csv")
    output_file = os.path.join(local_raw_data_path, "raw.csv")

    if check_existing_file(output_file):
        object_url = input_file
        print(f'downloading {input_file} as {os.path.basename(output_file)}')
        response = requests.get(object_url)
        if response.status_code == 200:
            content = response.text
            text_file = open(output_file, "wb")
            text_file.write(content.encode('utf-8'))
            text_file.close()
        else:
            print(f'Error accessing the object {input_file}:', response.status_code)

def main(local_raw_data_path="./data/raw_data", distant_folder_url="https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/"):
    """
    Upload data from S3 in ./data/raw_data
    """
    import_raw_data(local_raw_data_path, distant_folder_url)
    logger = logging.getLogger(__name__)
    logger.info("Raw dataset created")

if __name__=='__main__':
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    main()



