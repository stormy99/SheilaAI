#!/bin/bash

### INTRODUCTION
echo "=================================================================="
echo "A voice-activated smart-home assistant utilising a microcontroller"
echo "Prepared and written by Brandon Normington (ID: 10614655)"
echo "=================================================================="


### PRE-REQUISITES
echo "Checking if PyCharm is present.."
if dpkg-query -l | grep pycharm-professional || pycharm-community
then
	echo "PyCharm wasn't found! Installing.."
	sudo snap install pycharm-community --classic
fi
echo "------------------------------------------------------------------"


echo "Checking Python and Pip version.."
python3 --version && pip --version
echo "Checking if Pip has an available update.."
python3 -m pip install --upgrade pip
echo "------------------------------------------------------------------"


echo "Globally installing TF-GPU to configure for compute.."
python3 -m pip install tensorflow-gpu
echo "------------------------------------------------------------------"


echo "Installing miscellaneous library requirements.."
sudo apt-get install git tree curl jq xxd ffmpeg sox
echo "=================================================================="


### ENVIRONMENT SETUP
echo "Creating/navigating to a directory to house Sheila.."
cd ~/Desktop && mkdir sheila
echo "------------------------------------------------------------------"


echo "Cloning the Sheila repo from Git.."
git --version
git clone https://github.com/stormy99/sheila.git
cd ~/Desktop/sheila
echo "------------------------------------------------------------------"


echo "Now you need to download the matching Nvidia CUDAnn with the GPU driver and patch it (TF-GPU)"
echo "=================================================================="

