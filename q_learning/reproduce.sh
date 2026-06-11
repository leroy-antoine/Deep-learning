#!/bin/bash

echo "Training model..."
python3 train.py

echo "Running evaluation..."
python3 eval.py

echo "Done."