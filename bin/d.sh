#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../app

git clone https://github.com/nicolasfauchereau/Auckland_Cycling.git
cd Auckland_Cycling
git archive HEAD --format=tar.gz data > ../data.tar.gz
cd -
tar xzf data.tar.gz

rm -rf Auckland_Cycling data.tar.gz
