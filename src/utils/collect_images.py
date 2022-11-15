from bs4 import BeautifulSoup
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



def collect_fake_images(params):
    # avoids SSL certificate issue with website
    ssl._create_default_https_context = ssl._create_unverified_context

    os.chdir(params.data_collection.fake_image_dir)

    # set URL and objects to access website
    URL = params.data_collection.target_website
    page = requests.get(URL, verify=False)
    soup = BeautifulSoup(page.content, "html.parser")
    options = webdriver.ChromeOptions()
    options.add_argument('ignore-certificate-errors')
    chromedriver_location = params.data_collection.chromedriver_location
    driver = webdriver.Chrome(executable_path =r'chromedriver_location', chrome_options=options)

    # recursively access website and save images 
    driver.get("URL")
    scraping_attempts = params.data_collection.scraping_attempts

    for counter in tqdm(range(scraping_attempts)):
        try: 
            driver.get(URL)
            image = driver.find_element_by_id('avatar').get_attribute("src")
            file_name= re.match(r".*?r-(.*)" ,image).group(1)
            response = requests.get(image)
            if response.status_code==200:
                urllib.request.urlretrieve(image, file_name)
                file_name_list.append(file_name)
            else:
                print(response)
        finally:
            continue

def md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def deduplicate_fake_images(params):
    
    os.chdir(params.data_collection.fake_image_dir)
    dir_contents = os.listdir()
    filename_list, hash_list, size_list = [], [], []

    # loop through directory to gather features of each file
    for file in tqdm(dir_contents):
        size_list.append(os.path.getsize(file))
        filename_list.append(file)
        hash_list.append(md5_checksum(file))
    dupes = (pd.DataFrame(hash_list).value_counts()>1).sum()
    total_files = len(hash_list)
    percent_dupe = dupes/total_files
    print(f'Total Files {total_files}')

    # Hash metrics
    hash_list_df = pd.DataFrame(hash_list)
    hash_val_counts = hash_list_df.value_counts()
    unique_hash = len(hash_val_counts[hash_val_counts==1])
    not_unique_hash = len(hash_val_counts[hash_val_counts>1])
    total_unique_hash = unique_hash+not_unique_hash
    
    # Size metrics
    size_list_df = pd.DataFrame(hash_list)
    size_val_counts = size_list_df.value_counts()
    unique_size = len(size_val_counts[size_val_counts==1])
    not_unique_size = len(size_val_counts[size_val_counts>1])
    total_unique_size = unique_size+not_unique_size

    # Check for false positives: if hash are duplicate but file sizes are different
    file_df = pd.DataFrame([pd.Series(hash_list, name='hash'), pd.Series(filename_list, name='file'), pd.Series(size_list, name = 'size')]).T
    grouped_files = file_df.groupby(['hash', 'size']).count().sort_values(by='file')
    unique_hash_and_size = len(grouped_files['file'][grouped_files['file']==1])
    no_false_positives = unique_hash==unique_hash_and_size

    print(f'Unique hash {unique_hash}, Not Unique hash {not_unique_hash}, Total unique hash {total_unique_hash}')
    print(f'Unique size {unique_size}, Not Unique size {not_unique_size}, Total unique size {total_unique_size}')
    print(f'Unique percent hash {round((total_unique_hash/total_files)*100, 3)}%')
    print(f'Unique percent size {round((total_unique_size/total_files)*100, 3)}%')
    print(f'Currently no hash/size false duplicates identified? {no_false_positives}')
    

    # identify one version of duplicate files to keep and drop the rest
    # HASH
    hash_dupe_df = file_df.drop_duplicates(subset = ['hash'],
                                          ignore_index=True,
                                          keep = 'first'
                                    ).reset_index(drop = True)
    remove_these_df = file_df['file'][~file_df['file'].isin(hash_dupe_df['file'])]

    for file in remove_these_df:
        os.remove(file)

    #FILE SIZE    
    size_dupe_df = file_df.drop_duplicates(subset = ['size'],
                                          ignore_index=True,
                                          keep = 'first'
                                    ).reset_index(drop = True)
    remove_these_df = file_df['file'][~file_df['file'].isin(size_dupe_df['file'])]

    for file in remove_these_df:
        os.remove(file)

