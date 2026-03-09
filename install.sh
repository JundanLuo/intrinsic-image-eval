# conda create -n iid_eval python=3.8 -y
# conda activate iid_eval
conda install pytorch::pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
pip install opencv-python==4.7.0.72
pip install matplotlib==3.7.1
conda install scipy==1.10.1 scikit-image==0.19.3 numpy==1.23.5
pip install kornia==0.6.12
#pip install kornia[x]
conda install h5py==3.7.0
pip install pillow==10.4.0

