#!/bin/bash

datasets_dir='datasets/'

mkdir -p $datasets_dir/mnist

if ! [ -e $datasets_dir/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P $datasets_dir/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d $datasets_dir/mnist/train-images-idx3-ubyte.gz

if ! [ -e $datasets_dir/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P $datasets_dir/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d $datasets_dir/mnist/train-labels-idx1-ubyte.gz

if ! [ -e $datasets_dir/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P $datasets_dir/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d $datasets_dir/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e $datasets_dir/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P $datasets_dir/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d $datasets_dir/mnist/t10k-labels-idx1-ubyte.gz
