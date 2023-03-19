.DEFAULT_GOAL := help
.PHONY: help install clean format eval-model1 eval-model2 labels-distribution

## display help message
help:
	@awk '/^##.*$$/,/^[~\/\.0-9a-zA-Z_-]+:/' $(MAKEFILE_LIST) | awk '!(NR%2){print $$0p}{p=$$0}' | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort

venv := .venv
pip := $(venv)/bin/pip

$(pip):
	# create empty virtual environment containing pip
	$(if $(value VIRTUAL_ENV),$(error Cannot create a virtual environment when running in a virtual environment. Please deactivate $(VIRTUAL_ENV)),)
	python3.10 -m venv --clear $(venv)
	$(pip) install --upgrade pip wheel

## create virtual environment and install requirements
install: $(pip)
	$(pip) install -r requirements.txt

## remove the project virtual environment and artefacts
clean:
	rm -rf $(venv) data model1 model2

## format source code
format: $(venv)
	$(venv)/bin/black src/

## test source code
test: $(venv) data
	$(venv)/bin/pytest src/test.py -W ignore::DeprecationWarning

## download and unzip data
data:
	rm -rf data
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip -O uci.zip
	unzip uci.zip -d data/
	rm uci.zip

## train deep learning model
model1: $(venv) data
	time $(venv)/bin/python3 src/train.py deep-learning

## train non deep learning model
model2: $(venv) data
	time $(venv)/bin/python src/train.py non-deep-learning

## evaluate deep learning model
eval-model1: $(venv) data model1
	time $(venv)/bin/python src/evaluate.py deep-learning

## evaluate non deep learning model
eval-model2: $(venv) data model2
	time $(venv)/bin/python src/evaluate.py non-deep-learning

## check activity classes distribution
labels-distribution: data
	cat "data/UCI HAR Dataset/train/y_train.txt" | sort | uniq -c

