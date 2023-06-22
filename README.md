# htest

English | [日本語](./README_JA.md)


# usage

# Geting started  
The programming language is python, please install python environment before using it,  
conda with python 3.9+ is preferred, but other python envs are also fine.  

Firstly You could create an env with the command like below or using your own env:  
```bash 
conda create --name=htest python=3.9   
```

after created it successfully, you could activate it with this command:  
```bash
conda activate htest
```

and then install the necessary packages(except for pytorch) with the command: ``` pip install [package name]```:  
 ```bash
pandas  
scikit-learn  
torch_optimizer  
tensorboard  
```

or simply install all the above needed packages with the command: '```pip install -r requirements.txt```   

for pytorch installation, please refer to:  
https://pytorch.org/get-started/locally/  


# How to run

For privacy, I didn't upload the excel file, but for simply testing, please rename the test excel file into "MealAnalysis(2017).xlsx", and put into the same folder with this file.   

### for machine learning approach  

run with this command:   
```python ml.py```  

you will see your accuracy, if with this line:  
https://github.com/anguoyang/htest/blob/main/ml.py#L10  
The accuracy will be 1.0 which means 100%, if comment this line, then accuracy is around 80%    

### for neural network approach  

run with this command:  
```python nn.py```  

you will see your accuracy, if with this line:  
https://github.com/anguoyang/htest/blob/main/nn.py#L16  
The accuracy will be 1.0 which means 100%, if comment this line, then accuracy is around 80%   

if you want to check the training history, please use this command:  
```tensorboard --logdir "./runs"``` 
and open your browser and input this URL:    
```http://localhost:6006/```     
you will see your training loss history and accuracy evaluation history.  







