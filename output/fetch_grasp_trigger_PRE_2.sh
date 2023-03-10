#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=grasp_trigger_PRE_2.tar.gz
ID=16QjJeo94S3kqTualpzgTs5xhPOgH3EQm
CHECKSUM=5c74f6324115723167999e132a5124e4

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

echo "Downloading grasp_trigger_PRE_2 (102M)..."

gdown $ID

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
