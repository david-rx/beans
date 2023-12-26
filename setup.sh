#get repo
git clone https://github.com/david-rx/beans.git
#install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
source Anaconda3-2022.10-Linux-x86_64.sh
# install dependencies
conda create -n beans python=3.8 pytorch cudatoolkit=11.3 torchvision torchaudio cudnn -c pytorch -c conda-forge
pip install -r requirements.txt
pip install -e .
pip install kaggle
sudo apt-get install sox
sudo apt-get install libsox-fmt-mp3

# setup kaggle manually
