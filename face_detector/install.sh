#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate
pip3 install mlflow
pip3 install google_utils
pip3 install -qr requirements.txt 



