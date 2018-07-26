#!/usr/bin/env bash

if [ -d cifar-10-data ]; then
    rm -rf cifar-10-data
fi
mkdir cifar-10-data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz --strip-components=1 -C cifar-10-data
rm -f cifar-10-python.tar.gz
