# Congratulations!!
## You've just spotted your first deepfake
Okay probably not - they're everywhere. That's what this project is about. 
1. Using technology to spot deepfakes 
2. Learning how to spot them more easily without the aid of technology using explainable AI



![alt text](https://i.insider.com/5c6d85ca2628986f7f3a5d02?width=1000)


## TLDR: 
This project trains lots of different convolutional neural networks (CNNs) and tracks the experiments to find the best model to spot deepfakes - from scratch (without MLFlow or gridsearch). The best model is ![shared online](https://donovinemerson.com/?p=273) (manually for now). As an added bonus, this model will train *YOU* how **not** to not need it by explaining what it finds most important about the image for its prediction. 

There were four convolution layers in the best model. We see each one light up green where the model found the most important information (eyes, structural area, outlines, and teeth). Protip: the areas with the highest geometric variablility are where to look first.

![image](https://user-images.githubusercontent.com/87036676/216816048-caa696c7-1128-4d86-8a95-82f0bc1d2a01.png)

PS: Apologies to any visitors who are red/green colorblind! Unfortunately this color scheme was so much better than the other choices. 


## Diving Deeper:
That's not all that's been done in this project but we'll get into that, well... now.

Model training was conducted on ![StyleGAN generated images](https://arxiv.org/pdf/1812.04948.pdf) from http://this-person-does-not-exist.com. The benchmark "real" human pictures are from the ![Flicker Faces dataset FFHQ](https://github.com/NVlabs/ffhq-dataset). 
Performance is excellent for this dataset (F1 score >99%) but the zero shot cases from other known deepfake images do not fare so well. I'm working to make the 55GB dataset available but have not successed - yet!

## Repo Contents
Contained in this project are two applications 
1. Webscraper/webcrawler that gathers deefakes to train models
2. A custom OS independint experiment tracker
    - Ingests a configuration file to manage artifacts, the data pipeline, and model parameters - **params.yaml**
    - Permutes model parameters (cartesian product) to create the list of experiments
    - Trains and evaluates each of the models
        -  For each model, a new folder is created in the artifacts directory to store plots and saved model files (pb, hdf5, h5)
        -  After all models are trained, an aggregate DataFrame with details of each experiment (parameters and results) is stored in the main artifacts directory
    

## NOT Contained in this Repo
This model is available to interact with on my ![personal website](https://donovinemerson.com/?p=273) and is served using Streamlit - please play with it and let me know what you think! Link to that repo is ![here](https://github.com/doemerson123/fake-detector-api)


## Project Tree:

Most logic resides in the utils files. The deepfake scraper and testing folders are stand alone however model_training.py in the root directory relies on all the files in the utils folder. 

    |- deepfake_scraper
        |- data_collection.py ****Entrypoint for webscraping
        |- webscraping_util.py
    |- test
        |- Fake
            |- test_Fake 4.jpg
        |- conftest.py
        |- test_data_pipeline.py
        |- test_modeling_utils.py
        |- test_params.yaml
    |- utils
        |- custom_metrics_utils.py
        |- data_pipeline_utils.py
        |- modeling utils.py
        |- plot_metrics_utils.py
    |- model_training.py ****Entrypoint for model training
    |- params.yaml
    |- Dockerfile
    |- Makefile
    |- environment.yml
    |- requirements.txt
    |- README.md


## Is this thing on????
Since all configuration for the app occurs in params.yaml, entry points do not have a CLI. Executing either of the two .py files/modules in the terminal will trigger the logic: `python model_training.py` or `python data_collection.py` 

Test strategy is implemented using pytest. All code and supporting files reside in the test folder. Calling pytest in your favorite way will bring it to life.



## How to kickoff training
Edit params.yaml to desired parameter set, then run ```python model_training.py``` in the root of the repo

## How to kickoff webscraping
Edit params.yaml to desired parameter set, then run ```python deepfake\data_collection.py``` in the root of the repo using the appropriate slash for your system

high level overview of scraper and how it works (to come)

links to docs directory technical details for all utils (to come)


```
