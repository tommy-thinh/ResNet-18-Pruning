#!/bin/bash

watch -t -n 1 '
echo "===== GPU =====";
gpustat -P -up;
echo;
echo "===== RAM =====";
free -h;
echo;
echo "===== Disk =====";
df -h .;
'