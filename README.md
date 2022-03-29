# DSAI-HW1.2022
### Environment
* excute 
```bash=
$ py app.py --training "past.csv" --output submission.csv
```
* requrements
        (python version: 3.9.5)
```
numpy==1.19.5
pandas==1.4.1
scikit-learn==1.0.2
tensorflow==2.6.0
```
### 程式開發
* 使用模型：
1. LSTM:
    
    * 一種RNN模型，RNN會將隱藏層的output存入memory，當下次input資料作訓練時，會加入上一次存在Memory中的值一起計算。
    * LSTM改善了RNN的一些問題，其中包含3個gate：Input Gate、Output Gate、Forget Gate
        * Input Gate: 某個neuron的output是否能被寫到memory cell裡面，並且只有在gate是開啟的時候，才能做動作。
        * Output Gate: 其他的neuron可不可以從這個memory cell裡面把值讀出來，並且只有在gate是開啟的時候，才能做動作。
        * Forget Gate: 是否要把memory cell過去記得的東西忘掉或是否要把過去記得的東西做一下format。
    * 實作： 
        
        1.對資料做正規化，使資料介於0~1
        
        2.將資料分成前一期間&後一期間(給訓練使用)
        
        3.LSTM模型建立與訓練(只有一層LTSM和一層output)
3. SARIMA：
    * SARIMA模型為考量到**季節性**，ARIMA model的一種進階應用。    
    ARIMA(p, d, q)=![](https://i.imgur.com/2m7w0ad.png)
    * ARIMA是一種AR與MA結合在一起的線性迴歸模型，可以直觀的解釋為兩部分：過去資料的加權平均(AR)與隨機誤差的加權平均(MA)。    
    * ARIMA中I所代表的意義：time series需要差分的次數
    * 資料分析與模型定階:
        * ADF Test：時間序列的預測中，需要先檢定資料是否為定態，而ADF test是其中的一種方法，對資料建立迴歸式，檢查是否有unit root存在。
        * 第一次的檢測失敗(ADF Test的p-value > 0.05)
        * 代表Training Data需要一階差分→差分後通過ADF Test
        * ACF, PACF Plot：檢驗資料期數間彼此相關係數
        ![](https://i.imgur.com/OgYQbLq.png)
        * 觀察相關性較高的期數協助SARIMA的定階 (理論上)
        * (實際上) 利用parameter list暴力法建模，選出AIC指標最低的模型
### Result
* 在程式中，我們以2022-03-14到2022-03-28做為測試資料，03-14之前作為訓練資料，在結果中LSTM的RMSE為522，而SARIMA的RMSE為701
* 實作後發現LSTM的結果比較準確，所以在此筆資料上，我們使用LSTM的模型來做預測。
* 推論LSTM表現較佳可能的原因：在迴歸模型中，多處理具有線性關係的序列，而台電備轉容量為非線性，另外根據PACF與ACF Plot，可以觀察到序列期數之間沒有太大的相關性，故使用迴歸模型可能在表現上並不優異。