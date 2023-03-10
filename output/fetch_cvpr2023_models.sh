#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=cvpr2023_models.tar.gz
ID=1QAeMoF9N42DronDV1_SlUgwO1iexwCuG
CHECKSUM=b3730d6c5b81cd57aa16e7541abe1bcc

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading cvpr2023 models (233M)..."

gdown $ID

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
