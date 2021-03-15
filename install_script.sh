#!/bin/sh
git clone https://github.com/thorbenlouw/learning-amg.git
cd learning-amg/ || exit

sudo apt-get install octave liboctave-dev
octave --no-gui --eval "pkg install -forge statistics io"

pip install virtualenv
virtualenv --system-site-packages ./venv
. venv/bin/activate
pip install dm-sonnet==1.35
pip install tqdm
pip install oct2py
pip install matplotlib fire
pip install meshpy
pip install pyamg graph-nets==1.1.0
pip install tensorflow-probability==0.7.0

#  python train.py -config GRAPH_LAPLACIAN_TRAIN -eval-config GRAPH_LAPLACIAN_EVAL
