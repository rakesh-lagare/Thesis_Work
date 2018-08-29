#! /bin/sh -f
# or, adjust shell appropriately..

CURDIR=`pwd`
cd data
java -Dsun.java2d.noddraw=true -jar ../build/tq.jar -d0gt

cd ..
