#!/bin/bash
gdown --folder 1UySZdfOtuf00I5xrr98j3zRsSSfICTxe
cd raw
unzip broker.zip
unzip market.zip
unzip global.zip 
rm -rf __MACOSX
rm -rf broker.zip
rm -rf market.zip
rm -rf global.zip
cd ..
rm -rf __MACOSX

