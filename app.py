import pandas as pd  # pandas庫
import numpy as np  # numpy庫
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import csv


def OutputFile(filename, predict_):
    file = open(filename, mode='w', newline='')
    r = csv.writer(file)
    predict_ = np.array(predict_)
    predict_ = predict_.flatten()
    date = ['20220330', '20220401', '20220402', '20220403', '20220404',
            '20220405', '20220406', '20220407', '20220408', '20220409',
            '20220410', '20220411', '20220412', '20220413']
    for cnt in range(0, 14):
        r.writerow([date[cnt], predict_[cnt+2]])
        cnt += 1
        # print(cnt)
    file.close()


def prediction(num_predict, model, dataset):
    look_back = 14
    prediction_list = dataset[-look_back:]
    for _ in range(num_predict):
        x = prediction_list[-look_back:]
        x = np.reshape(x, (1, 1, look_back))
        # print(x)
        out = model.predict(x)
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list


# 產生 (X, Y) 資料集, Y 是下一期的乘客數
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def Train(filename, output):
    # 載入訓練資料
    dataframe = pd.read_csv(filename, usecols=[1])
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # 正規化(normalize) 資料，使資料值介於[0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset))
    train = dataset[0:train_size, :]

    # 產生 (X, Y) 資料集, Y 是下一期的乘客數(reshape into X=t and Y=t+1)
    look_back = 14
    trainX, trainY = create_dataset(train, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    # 建立及訓練 LSTM 模型
    model = Sequential()
    model.add(LSTM(4,  input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
    model.save('model.h5')
    model = load_model("model.h5")

    # my predict function
    num_predict = 15
    predict_ = prediction(num_predict, model, dataset)
    predict_ = np.reshape(predict_, (len(predict_), 1))

    # 預測
    trainPredict = model.predict(trainX)
    trainPredict = np.concatenate([trainPredict, predict_])

    # 回復預測資料值為原始數據的規模
    trainPredict = scaler.inverse_transform(trainPredict)
    predict_ = scaler.inverse_transform(predict_)
    OutputFile(output, predict_)


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    print(args.training)
    Train(args.training, args.output)
