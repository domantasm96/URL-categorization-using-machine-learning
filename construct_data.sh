#!/bin/bash
echo "Executing construct_features.py"
python3 01_construct_features.py
echo "Executing construct_models.py"
python3 02_construct_models.py
echo "Executing train_models.py"
python3 03_train_models.py
