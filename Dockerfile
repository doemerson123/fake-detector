#FROM frolvlad/alpine-miniconda3
FROM ubuntu:18.04

#RUN apk update \
#    && apk install python3.8 python3-pip  --upgrade pip -y

COPY . .

# ALPINE
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"


# ALPINE APK\

#RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
#RUN python3 -m ensurepip
#RUN pip3 install --no-cache --upgrade pip setuptools

#RUN apt-get update && apt-get install -y python=3.7
RUN apt-get update && apt-get install -y wget
RUN apt-get install wget  && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    # tensorflow image didn't like this
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -yes \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Create the environment:
##COPY environment.yml . was working okay before copy .
#RUN conda update conda 
RUN conda env create -f py36.yml
RUN pip install tensorflow-gpu=2.3
# Make RUN commands use the new environment:
#RUN echo "conda activate fake-detector" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]



#RUN echo "conda list"
CMD ["conda", "run", "-n", "fakedetector", "python3", "model_training.py"]