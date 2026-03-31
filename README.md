# NORC_rudn
A pipeline for histology image processing using Qupath and Python.

This repository provides an end-to-end workflow for segmenting tissue regions using DeepCell and QuPath.
It includes scripts for tiling annotated regions, running deep learning-based segmentation, and importing results back into QuPath for downstream analysis.
Workflow Overview
Tile Export: Run segment_tiles.groovy in QuPath to generate TIFF patches from selected annotations. It automatically calls for nondummy_segmentation.py to process the images
NOTE! Setup Anaconda path to run python directly from segment_tiles.groove.

Segmentation: Use nondummy_segmentation.py to run a DeepCell model on the exported TIFFs.

Annotation export: export_data_for_cellsighter.groovy allows for exporting data from QuPath in CellSighter-digestable format 

Fine-tuning: A dedicated script is provided for model refinement on custom data.

Installation
QuPath Setup
Tested on QuPath v0.6
Place the scripts from scripts/qupath/ into your QuPath project's scripts directory.

NOTE! You have to use Projects instead of images to ensure everyhting runs smoothly. 

Python Setup
Install the required dependencies for DeepCell:
bash
pip install -r requirements.txt
Use code with caution.



Usage
1. Exporting Tiles and segmenting (QuPath)
Open your project in QuPath, select an annotation, edit segment_tiles.groovy to point to your Anaconda and required venv and run segment_tiles.groovy. This will save TIFF tiles to a designated output folder.

4. Fine-Tuning
To retrain or fine-tune the model with your own labeled data:
bash
python scripts/deepcell/train_model.py --data_path ./training_data
Use code with caution.

License
This project is licensed under the MIT License. (TO ADD...) 
