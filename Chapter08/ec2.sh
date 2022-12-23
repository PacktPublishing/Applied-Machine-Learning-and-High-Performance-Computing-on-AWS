#!/bin/bash
aws s3 cp s3://green-grass-v2/scripts/ /home/ec2-user --recursive
sleep 30
process_id=$!
wait $process_id
sudo yum update -y 
sudo yum install docker -y
sudo yum install python-pip -y
sudo pip3 install boto3
sudo pip3 install requests
cd /home/ec2-user
python3 getResourceTempCredentials.py
sudo service docker start
sudo usermod -a -G docker ec2-user
docker build -t "aws-iot-greensgrass:2.5" ./
chmod +x dockerRun.sh
./dockerRun.sh
