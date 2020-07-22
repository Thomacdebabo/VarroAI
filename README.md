# AI based detection of dead varroa

Beekeepers nowadays have to worry a lot about a species of parasitic mites called varroa. A beekeeper has to monitor the amount of varroa in each beehive and has to count how many dead ones there are.

This project aims to eliminate the need to manually count the amount of varroa for each beehive. Instead we want to use classical computer vision algorithms combined with machine learning.

## Training the model
I made a small GUI so we can track the training progress. It provides a small selection of filters and some examples. It also allows you to save your models and load already trained ones.

![GUI](https://github.com/Thomacdebabo/VarroAI/blob/master/Train_GUI.JPG)

you can use the train.py script which only contains:
```python
import Source.GUI_Train as GUI

GUI.start_training_GUI(data_path=r"PATH/TO/DATA")
```
## Count Varroa
To actually count the varroa in a picture you can use the batchprocessing command which you have to provide with a path to a directory which contains all pictures which should be processed. This process right now will just use a already trained model in the Model directory.
```python
import Source.Utils as utils

utils.batchProcessing(r"Path/to/Images")
```

This process will just output the results in the console. Will probably ad csv, json or txt files for that at some point.

## Extend the Dataset
I also provided a small script which allows you to easily classify a load of data in a short amount of time.

```python
import Source.Utils as utils

img_dir = r"IMAGE PATH"
dataset_dir = r"Data"

utils.Img_to_Dataset(img_dir, dataset_dir)
```
This script allows you to classify possible varroa matches in a picture. It will show you a sample and you have to press the right key:
- "a" to label a sample as varroa,  
- "d" to tag it as a false positive,
- any other key to skip the sample

The labeled images are stored at the location: dataset_dir


## Requirements
This code works on python 3.7 with:
- tensorflow 2.0 (with keras)
- numpy
- matplotlib
- PyQt5
- pyqtgraph
- opencv
