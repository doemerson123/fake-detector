# Congratulations!!
## You've just spotted your first deepfake
Okay probably not - they're everywhere. That's what this project is about. 
1. Using technology to spot deepfakes 
2. Learning how to spot them without the aid of technology using explainable AI



![alt text](https://i.insider.com/5c6d85ca2628986f7f3a5d02?width=1000)


## TLDR: 
This project trains lots of different convolutional neural networks (CNNs) and tracks the experiments to find the best model to spot deepfakes - from scratch (no MLFlow or gridsearch). The best model is ![shared online](https://donovinemerson.com/?p=273) (manually for now). As an added bonus, this model will train you how *NOT* to not need it by explaining what it finds most important about the image for its prediction. 

There were four convolution layers in the best model. We see each one light up green where the model found the most important information (eyes, structural area, outlines, and teeth). Protip: the areas with the highest geometric variablility are where to look first.

![image](https://user-images.githubusercontent.com/87036676/216816048-caa696c7-1128-4d86-8a95-82f0bc1d2a01.png)

PS: Apologies to any visitors who are red/green colorblind! Unfortunately this color scheme was so much better than the other choices. 


## High Level:
That's not all that's been done in this project but we'll get into that, well... now.

Model training was conducted on ![StyleGAN generated images](https://arxiv.org/pdf/1812.04948.pdf) from http://this-person-does-not-exist.com. The benchmark "real" human pictures are from the ![Flicker Faces dataset FFHQ](https://github.com/NVlabs/ffhq-dataset). 
Performance is excellent for this dataset (F1 score >99%) but the zero shot cases from other known deepfake images do not fare so well. I'm working to make the 55GB dataset available but have not successed - yet!

## Repo Contents
Contained in this project are two applications 
1. Webscraper/webcrawler that pulls down all the deepfakes we need from 
2. A custom training harness that
    - Ingests a yaml config file with model paramters(params.yaml)
    - 



Classifies an image of a person as real or fake (GAN generated)

Project tree 

Images of streamlit 

additional detail - slides

how to interact with API



how to kickoff training

high level overview of scraper and how it works

links to docs directory technical details for all utils

