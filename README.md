# drugGraphNet
Thesis main project 
Here we develop novel models that build on variational graph auto-encoders and can integratediverse types of data to provide high quality predictions of genetic interactions, cell line dependencies anddrug sensitivities.

<!-- ## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup) -->
___
## Requirements:

You can also installed the python environment by running:
   `$ pip install -r requirements.txt`

___
 ## How to run:
 Follow the following steps:
 
 Step1: Select the prediction task and the required dataset.
 
 Step2: Decompress the required data file, for example:
 
 `unzip data_sl.zip`

  Step3: Run the required file based on your task prediction:
  * For training Genetic Interations:
  `$ python main_gi.py --dataset selected_dataset`
  
  * For training Cancer Dependency:
  `$ python main_dependency.py --dataset selected_dataset`
  
  * For training Drug Sensitivity:
  `$ python main_drug.py --dataset selected_dataset`
 
 
