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

for pytorch installation, please refer to:  
https://pytorch.org/get-started/locally/  

***If there are still other modules need to be installed or other problems, please install them accordingly(follow the hints), or open an issue, thank you.***  

# How to run

I didn't upload the excel file because of privacy, but for simply testing, please rename the test excel file into "MealAnalysis(2017).xlsx", and put into the same folder with this file.   

### for machine learning approach  

run with this command:   
```python ml.py```  
you will see the evaluation accuracy regarding the test data.

by the way, if with this line:  
https://github.com/anguoyang/htest/blob/main/ml.py#L10  
The accuracy will be 1.0 which means 100%, if comment this line, then accuracy is around 80%    

### for neural network approach  

run with this command:  
```python nn.py```  
you will see the training status as well as the evaluation accuracy regarding the test data. 

by the way, if with this line:  
https://github.com/anguoyang/htest/blob/main/nn.py#L16  
The accuracy will be 1.0 which means 100%, if comment this line, then accuracy is around 80%   

if you want to check the training history, please use this command:  
```tensorboard --logdir "./runs"``` 
and open your browser and input this URL:    
```http://localhost:6006/```     
you will see your training loss history and accuracy evaluation history.  

***Please note that we decrease the target with 1 to let it start from 0 [0,3], so in real senarios, please add it back to the output to make sure the final output range [1,4] ***





