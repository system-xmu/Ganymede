#!/bin/bash -x

sudo make bindeb-pkg -j70
cd ..
sudo dpkg -i *.deb
