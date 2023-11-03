#!/bin/bash
# These codes are licensed under MIT License.
# Copyright (C) 2023 yskn67
# https://github.com/yskn67/benzaiten/blob/2nd/LICENSE

set -euxo pipefail

# Git
cp .devcontainer/.gitconfig ~/.gitconfig
git config --global --add safe.directory /workspace
git config --global user.email $GIT_EMAIL
git config --global user.name $GIT_USERNAME