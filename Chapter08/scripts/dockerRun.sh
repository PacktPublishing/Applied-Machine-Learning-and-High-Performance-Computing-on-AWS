#!/bin/bash
docker run --rm --init -it --name aws-iot-greengrass \
 -v "$(pwd)":/root/.aws/:ro \
 --env-file env \
 -p 8883 \
 aws-iot-greensgrass:2.5