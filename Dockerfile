FROM tensorflow/tensorflow:2.13.0-gpu
# this contains CUDA 11.0 and CUDNN 8.X.X

RUN apt-get update -y
RUN apt-get install sudo -y
RUN apt-get install git -y

# Creating a non root user
RUN useradd -ms /bin/bash ubuntu

# switch into the jax user
USER ubuntu
# add the user's .local/bin to path so that our --user pip installed packages will appear
ENV PATH=/home/ubuntu/.local/bin:$PATH
# install python packages via pip
# for CUDA support, we have to specify a particular jaxlib version
# we also need to specify the particular cuda version
RUN pip3 install --user \
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    pip install jaxlib[cuda11_pip]

CMD /bin/bash


