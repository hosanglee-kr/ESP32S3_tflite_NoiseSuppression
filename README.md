# PlatformIO CI/CD template

[![Test](https://github.com/256dpi/arduino-mqtt/actions/workflows/test.yml/badge.svg)](https://github.com/hosanglee-kr/PlatformIO_Template_V001/actions/workflows/test.yml)
[![GitHub release](https://img.shields.io/github/release/256dpi/arduino-mqtt.svg)](https://github.com/hosanglee-kr/PlatformIO_Template_V001/releases)



This repository template contains all the necessary files to:

1. Do PlatformIO development with VSCode and remote development containers.
2. GitHub actions for CI/CD, including attaching firmware binaries automatically to every pull request
   and on every GitHub release.
3. Automatic version number assignment at build time. Local builds get version `0.0.1`, pull request
   builds get version `0.0.{pull request number}` and release builds get the version from the GitHub
   release tag.

## Using this repository template

Using this repository template is easy: just hit the _Use this template_ button and make a new repo. That's it!


## git version 확인 및 update
git --version
git update-git-for-windows


## shell script 실행 변경
git update-index --chmod=+x prepare_deploy.sh

## submodule add command
git submodule add https://github.com/pschatzmann/arduino-audio-tools.git lib/arduino-audio-tools
git submodule add https://github.com/bblanchon/ArduinoJson.git lib/ArduinoJson

## subModule 상태확인
git submodule status

## submodule update
# 방법 1
git submodule init
git submodule update

##git submodule update --remote

## Submodule 삭제
$ git submodule deinit ./lib/arduino-audio-tools/

