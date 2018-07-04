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
### Training ACC & LOSS
#### MLP Training ACC & LOSS
將train data用MLP Model訓練後，得到的準確率為99.5%，將測test data代入後，得到的準確率為98.11%  
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/MLP%20acc.PNG?raw=true "mlp acc")  
#### CNN Training ACC & LOSS
將train data用CNN Model訓練後，得到的準確率為99.01%，將測test data代入後，得到的準確率為99.12%  
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/CNN%20acc.PNG?raw=true "cnn acc")  
#### ACC & LOSS
|     | Train ACC | Train LOSS | Test ACC | 
|:---:|:---------:|:----------:|:--------:|
| MLP | 0.9950    | 0.0150     | 0.9811   |  
| CNN | 0.9901    | 0.0305     | 0.9912   |  
### Confusion Matrix
#### MLP
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/MLP%20confusion%20matrix.PNG?raw=true "mlp cm")  
#### CNN
![Image I](https://github.com/NdhuCarrey/ML2018-410321146-HW2/blob/master/result/CNN%20confusion%20matrix.PNG?raw=true "cnn cm")  
### Discussion
本次作業利用MLP及CNN的方式進行訓練，發現CNN的準確率稍微比MLP的準確率還高，訓練時間也比MLP還來的高許多。從Confusion Matrix中可發現其辨識結果，MLP對數字3的錯誤率較高，而CNN對數字8的錯誤率較高。  
這次作業練習，學習到利用MLP及CNN的實作方式，能夠更加了解兩者的訓練內容不同。在進行CNN練習時，因還不夠了解CNN的輸入及Model架構，加上CNN訓練時間較長，因此在練習中進行較長時間的偵錯，最後訓練完畢後，兩者的準確率都有98%以上，希望下次練習時，可以使用不同的方法，或是修改目前的model及一些參數，提高其準確率。
