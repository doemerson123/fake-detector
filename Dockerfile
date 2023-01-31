#FROM frolvlad/alpine-miniconda3
FROM tensorflow/tensorflow:2.5.1

#RUN apk update \
#    && apk install python3.8 python3-pip  --upgrade pip -y

COPY . .




#RUN apk update && apk add bash
#RUN apk add wget # && rm -rf /var/lib/apt/lists/*

# Create the environment:
##COPY environment.yml . was working okay before copy .
#RUN conda update conda 
#RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
#RUN echo "conda activate fake-detector" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]




RUN pip install -r requirements.txt
RUN python model_training.py
#CMD ["conda", "run", "-n", "fakedetector", "python3", "model_training.py"]