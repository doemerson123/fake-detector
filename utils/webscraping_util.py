import hashlib
import re
import ssl
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import ssl
from tqdm import tqdm
import urllib
from utils.load_params_util import load_params

params = load_params('params.yaml')

def collect_fake_images():
    '''
    Connects to target site and saves images to the fake image directory 
    params.yaml defines directory, iterations, target site, and element to save
    '''

    # alter chrome driver to avoid SSL certificate issue with target website 
    ssl._create_default_https_context = ssl._create_unverified_context
    options = webdriver.ChromeOptions()
    options.add_argument('ignore-certificate-errors')
    chromedriver_location = params.data_collection.chromedriver_location
    driver = webdriver.Chrome(executable_path = chromedriver_location, 
                              chrome_options = options)

    url = params.data_collection.target_website
    scraping_attempts = params.data_collection.scraping_attempts
    element_id = params.data_collection.element_id
    attribute_to_retreive = params.data_collection.attribute_to_retreive
    fake_image_dir = params.data_collection.fake_image_dir

    for counter in tqdm(range(scraping_attempts)):
        try: 
            driver.get(url)
            image = driver.find_element_by_id(element_id)\
                        .get_attribute(attribute_to_retreive)
            file_name= re.match(r".*?r-(.*)" ,image).group(1)
            response = requests.get(image)
            if response.status_code == 200:
                urllib.request.urlretrieve(image, fake_image_dir + file_name)
            else:
                print(response)
        finally:
            continue

def md5_checksum(fname):
    '''
    Returns MD5 checksum of a file
    '''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def deduplicate_fake_images():
    '''
    Uses MD5 remove any duplicate images in fake image directory. 
    MD5 is prone to false positives so file size is used as a secondary check 
    '''
     
    fake_image_dir = params.data_collection.fake_image_dir
    dir_contents = os.listdir(fake_image_dir)
 
    # loop through directory to calculate hash and get file size
    filename_list, hash_list, size_list = [], [], []  
    for file in tqdm(dir_contents):
        hash_list.append(md5_checksum(file))
        size_list.append(os.path.getsize(file))
        filename_list.append(file)

    total_files = len(hash_list)
    print(f'Total Files {total_files}')

    # define hash metrics
    hash_list_df = pd.DataFrame(hash_list)
    hash_val_counts = hash_list_df.value_counts()
    num_unique_hash = len(hash_val_counts[hash_val_counts==1])
    num_not_unique_hash = len(hash_val_counts[hash_val_counts>1])
    total_unique_hash = num_unique_hash + num_not_unique_hash
    
    # define size metrics
    size_list_df = pd.DataFrame(hash_list)
    size_val_counts = size_list_df.value_counts()
    num_unique_size = len(size_val_counts[size_val_counts==1])
    num_not_unique_size = len(size_val_counts[size_val_counts>1])
    total_unique_size = num_unique_size+num_not_unique_size

    # consilidate metrics into df then count group for each file
    file_df = pd.DataFrame([pd.Series(hash_list, name='hash'), 
                            pd.Series(filename_list, name='file'), 
                            pd.Series(size_list, name = 'size')]).T
    grouped_files = file_df.groupby(['hash', 'size'])\
                            .count()\
                            .sort_values(by='file')

    # compare count of unique files using hash only to the 
    # count using both hash and file size. If equal, no MD5 false positives
    unique_hash_and_size = len(grouped_files['file'][grouped_files['file']==1])
    false_positives_bool = num_unique_hash==unique_hash_and_size

    print(f'Unique hash {num_unique_hash}, Not Unique hash {num_not_unique_hash}, Total unique hash {total_unique_hash}')
    print(f'Unique size {num_unique_size}, Not Unique size {num_not_unique_size}, Total unique size {total_unique_size}')
    print(f'Unique percent hash {round((total_unique_hash/total_files)*100, 3)}%')
    print(f'Unique percent size {round((total_unique_size/total_files)*100, 3)}%')
    print(f'Currently no hash/size false duplicates identified? {false_positives_bool}')  

    # identify one version of duplicate files to keep and drop the rest 
    # using hash
    hash_dupe_df = file_df.drop_duplicates(subset = ['hash'],
                                          ignore_index=True,
                                          keep = 'first'
                                            ).reset_index(drop = True)
    remove_hash_df = file_df['file'][~file_df['file']\
                                    .isin(hash_dupe_df['file'])]

    # using file size   
    size_dupe_df = file_df.drop_duplicates(subset = ['size'],
                                          ignore_index=True,
                                          keep = 'first'
                                          ).reset_index(drop = True)
    remove_size_df = file_df['file'][~file_df['file']\
                                    .isin(size_dupe_df['file'])]
    for file in remove_hash_df:
        os.remove(file)
    for file in remove_size_df:
        os.remove(file)

