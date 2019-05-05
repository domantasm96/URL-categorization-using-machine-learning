#!/bin/bash
echo "Executing construct_features.py"
python3 construct_features.py
echo "Executing construct_models.py"
python3 construct_models.py
echo "Executing train_models.py"
python3 train_models.py
