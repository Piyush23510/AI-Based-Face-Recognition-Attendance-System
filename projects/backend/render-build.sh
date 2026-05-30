#!/usr/bin/env bash

apt-get update
apt-get install -y ffmpeg pkg-config

pip install -r requirements.txt
