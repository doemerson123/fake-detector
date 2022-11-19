# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install python3.8 python3-pip -y



# Install pip requirements
COPY requirements.txt .
RUN apt-get gssapi=1.6 -y

RUN KRB5_KRB5CONFIG="$( which krb5-config )" python setup.py bdist_wheel
RUN pip install krb5-config --cflags krb5
RUN apt-get install libkrb5-dev gcc krb5-config -y 
RUN pip install -r requirements.txt


RUN apt-get install libxext6 libsm6 -y 
RUN apt-get -qq install libgl1 -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata libgtk2.0 ffmpeg 

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser


CMD streamlit run api/api.py


