FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive 

# Install base dependencies
RUN apt update && apt install -y curl python3 python3-pip python3-setuptools iputils-ping
#TZ=Etc/UTC apt-get -y install tzdata

# Install pip packages
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt


WORKDIR /ei
COPY . ./ 
ENTRYPOINT ["python3", "-u", "app.py"]