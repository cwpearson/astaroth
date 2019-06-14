#!/bin/bash

#gm convert -delay 40 colden_*.png colden.gif

DATE=`date '+%Y_%m_%d_%H_%M'`

echo $DATE

gm convert -delay 15 $1_*.png $1_$DATE.gif
