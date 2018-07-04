# ML2018-410321146-HW2 
# Handwritten Character Recognition
## 作業內容
本次作業使用MLP、CNN的方式訓練手寫符號並做辨識，分析兩種方式的準確率及其差別，資料庫為MNIST。
## Result
### Data processing
#### MLP資料處理
in_train_MLP = in_train.reshape(60000, 784).astype('float32') #將資料存到np陣列中  
in_test_MLP = in_test.reshape(10000, 784).astype('float32')  
in_train_MLP_normalize = in_train_MLP / 255 #資料做normalize  
in_test_MLP_normalize = in_test_MLP / 255  
#### CNN資料處理
in_train_CNN = in_train.reshape(in_train.shape[0], 28, 28, 1).astype('float32') #將資料存到np陣列中(cnn輸入為4維陣列)  
in_test_CNN = in_test.reshape(in_test.shape[0], 28, 28, 1).astype('float32')  
in_train_CNN_normalize = in_train_CNN / 255 #資料做normalize  
in_test_CNN_normalize = in_test_CNN / 255  
### Model
#### MLP Model
MLP model 中間總共用了五層，包含輸入及輸出(輸出10)  
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/MLP%20model.PNG?raw=true "mlp model")  
#### CNN Model
CNN model 中間建立了兩層池化層,共有16個7x7維度的影像個神經元，因此總共有784個神經元，最後輸出為10個神經元  
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/CNN%20model.PNG?raw=true "cnn model")  
### Training ACC
#### MLP Training ACC
將train data用MLP Model訓練後，得到的準確率為99.5%，將測test data代入後，得到的準確率為98.11%  
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/MLP%20acc.PNG?raw=true "mlp acc")  
#### CNN Training ACC
將train data用CNN Model訓練後，得到的準確率為99.01%，將測test data代入後，得到的準確率為99.12%  
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/CNN%20acc.PNG?raw=true "cnn acc")  
### Confusion Matrix
#### MLP
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/MLP%20confusion%20matrix.PNG?raw=true "mlp cm")  
#### CNN
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/CNN%20confusion%20matrix.PNG?raw=true "cnn cm")  
### Discussion
