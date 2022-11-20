# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:11.6.2-base-ubuntu20.04
#FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install python3.8 python3-pip  --upgrade pip -y

#RUN apt-get update
#RUN apt-get install -y build-essentials
#RUN    apt-get install -y wget
#RUN apt-get clean
#RUN rm -rf /var/lib/apt/lists/*

# Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda

#RUN conda install -c conda-forge opencv
# Install pip requirements
COPY requirements.txt .
#RUN apt-get gssapi -y



##RUN pip install krb5-config --cflags krb5
#RUN KRB5_KRB5CONFIG="$( which krb5-config )" python setup.py bdist_wheel
##RUN apt-get install libkrb5-dev gcc krb5-config -y 
#RUN KRB5_KRB5CONFIG="$( which krb5-config )" python setup.py bdist_wheel

##RUN sudo ln -s /usr/bin/krb5-config /usr/bin/krb5-config
##RUN sudo ln -s /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2 /usr/lib/libgssapi_krb5.so
##RUN sudo apt-get install libkrb5-dev
##RUN sudo pip install gssapi -y



##RUN export PATH="$HOME/usr/bin/krb5-config:$PATH"
##RUN export PATH="$HOME/usr/bin/:$PATH"
##RUN export PATH="$HOME/usr/lib/:$PATH"
##RUN export PATH="$HOME/usr/lib/x86_64-linux-gnu/:$PATH"
##RUN export PATH="$HOME/bin/sh:$PATH"
#RUN apt-get update && apt-get install opencv-python-headless
#RUN pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl
##RUN python -m pip install opencv-python
RUN pip install -r requirements.txt
RUN pip3 install opencv-python --upgrade

##RUN apt-get install libxext6 
##RUN apt-get install libsm6 libGL.so.1 tzdata libgtk2.0 -y 
##RUN apt-get -qq install libgl1 libgl1-mesa-glx -y
#RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install  ffmpeg 



#RUN apt-get update && apt-get install -y python3-opencv
#RUN pip install opencv-python
##RUN apt-get install python3-opencv
##RUN ln -s /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1 /usr/lib/libGL.so.1

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser


CMD streamlit run api/api.py


