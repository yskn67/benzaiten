#!/bin/bash

set -euxo pipefail

# Git
cp .devcontainer/.gitconfig ~/.gitconfig
git config --global --add safe.directory /workspace
git config --global user.email $GIT_EMAIL
git config --global user.name $GIT_USERNAME