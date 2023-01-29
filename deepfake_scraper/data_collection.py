import os
from utils.data_pipeline_utils import load_params
from deepfake_scraper.webscraping_util import (
    collect_fake_images,
    deduplicate_fake_images,
)


def scrape_fake_images():
    """
    Scrapes GAN generated images and removes duplicates until the number of
    needed images is met.
    """
    params = load_params("params.yaml")

    total_needed_fake_images = params.data_collection.total_needed_fake_images
    total_saved_fake_images = 0

    while total_saved_fake_images <= total_needed_fake_images:
        collect_fake_images()
        deduplicate_fake_images()
        total_saved_fake_images = len(os.listdir(params.data_collection.fake_image_dir))
