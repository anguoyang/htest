# htest

# 使用法

# Geting started  
プログラミング言語はPythonです。使用する前にPython環境をインストールしてください。  
condaを使用し、Python 3.9以上の環境を作成することを推奨しますが、他のPython環境でも問題ありません。  

まず、以下のコマンドで環境を作成するか、独自の環境を使用してください:    
```bash 
conda create --name=htest python=3.9     
```

作成が成功したら、次のコマンドでアクティベートできます: 
``` 
conda activate htest
```

そして、以下のコマンドを使用して必要なパッケージ（pytorchを除く）をインストールします: ```pip install [パッケージ名]```:

pandas  
scikit-learn  
torch_optimizer  
tensorboard  

または、以下のコマンドを使用して上記の必要なパッケージをすべて一括でインストールします: ```pip install -r requirements.txt```

pytorchのインストールについては、次のリンクを参照してください:  
https://pytorch.org/get-started/locally/  

# 実行方法
プライバシーのため、Excelファイルはアップロードしていませんが、単純なテストの場合は、テストのExcelファイルを「MealAnalysis(2017).xlsx」という名前に変更し、このファイルと同じフォルダに配置してください。  

### 機械学習アプローチの場合  
次のコマンドで実行します:  
```python ml.py```
テストデータに関する評価精度が表示されます。

なお、この行がある場合、  
https://github.com/anguoyang/htest/blob/main/ml.py#L10  
正確性は1.0（100%）となりますが、この行をコメントアウトすると、正確性は約80%程度になります。

### ニューラルネットワークアプローチの場合  
次のコマンドで実行します:    
```python nn.py```  
トレーニングの状態と、テストデータに関する評価精度が表示されます。

なお、この行がある場合、  
https://github.com/anguoyang/htest/blob/main/nn.py#L16  
正確性は1.0（100%）となりますが、この行をコメントアウトすると、正確性は約80%程度になります。 

トレーニングの履歴を確認したい場合は、次のコマンドを使用してください:    
```tensorboard --logdir "./runs"```   
そして、ブラウザを開いて、以下のURLを入力してください:    
```http://localhost:6006/```      
トレーニングの損失履歴と正確性評価履歴が表示されます。  
