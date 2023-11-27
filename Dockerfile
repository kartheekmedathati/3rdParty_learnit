FROM tensorflow/tensorflow:2.13.0-gpu
# this contains CUDA 11.0 and CUDNN 8.X.X

RUN apt-get update -y
RUN apt-get install sudo -y
RUN apt-get install git -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Creating a non root user
RUN useradd -ms /bin/bash ubuntu

#  Add new user docker to sudo group
RUN adduser ubuntu sudo

# Ensure sudo group users are not 
# asked for a password when using 
# sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# switch into the jax user
USER ubuntu
# add the user's .local/bin to path so that our --user pip installed packages will appear
ENV PATH=/home/ubuntu/.local/bin:$PATH
# install python packages via pip
# for CUDA support, we have to specify a particular jaxlib version
# we also need to specify the particular cuda version


RUN pip install --user \
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    pip install matplotlib \
    pip install jaxlib[cuda11_pip] \
    pip install pillow \
    pip install optax \
    pip install dm-haiku==0.0.10 \
    pip install livelossplot \
    pip install tqdm \
    pip install pillow \
    pip install opencv-python \
    pip install tensorflow_datasets \
    pip install tqdm \
    pip install ipywidgets \
    pip install IProgress
 

CMD /bin/bash


