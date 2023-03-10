#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=cvpr2023_results.tar.gz
ID=1VazC1IBoaSk_OJUCeIAUdqRtJ1vSgw_1
CHECKSUM=fb455a34b5ec8403df2b6be378942092

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

echo "Downloading cvpr2023 results (17M)..."

gdown $ID

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
