.DEFAULT_GOAL := help
.PHONY: help install clean preprocess

## display help message
help:
	@awk '/^##.*$$/,/^[~\/\.0-9a-zA-Z_-]+:/' $(MAKEFILE_LIST) | awk '!(NR%2){print $$0p}{p=$$0}' | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort

venv := .venv
pip := $(venv)/bin/pip

$(pip):
	# create empty virtual environment containing pip
	$(if $(value VIRTUAL_ENV),$(error Cannot create a virtual environment when running in a virtual environment. Please deactivate $(VIRTUAL_ENV)),)
	python3 -m venv --clear $(venv)
	$(pip) install --upgrade pip wheel

$(venv): $(pip)
	$(pip) install -r requirements.txt

## create virtual environment and install requirements
install: $(venv)
	$(venv)/bin/pre-commit install

install-hooks: .git/hooks/pre-commit

## run pre-commit git hooks on all files
hooks: install-hooks $(venv)
	$(venv)/bin/pre-commit run --show-diff-on-failure --color=always --all-files --hook-stage push

## remove the virtual environment
clean:
	rm -rf $(venv)

## download and unzip data
download-data:
	rm -rf data
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip -O uci.zip
	unzip uci.zip -d data/
	rm uci.zip

## train
train: $(venv) data
	rm -rf keras_artefacts
	$(venv)/bin/python3 src/train.py

## evaluate
evaluate: $(venv) model
	$(venv)/bin/python3 src/evaluate.py

