import os
from utils.load_params_util import load_params
from utils.webscraping_util import collect_fake_images, deduplicate_fake_images

params = load_params("params.yaml")

total_needed_fake_images = params.data_collection.total_needed_fake_images
total_saved_fake_images = 0

while total_saved_fake_images <= total_needed_fake_images:
    collect_fake_images()
    deduplicate_fake_images()
    total_saved_fake_images = len(os.listdir(params.data_collection.fake_image_dir))
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                