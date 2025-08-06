from numpy import ndarray, array

def sequentialize(data: ndarray, lookback: int = 24, horizon: int = 24) -> (ndarray, ndarray):
    input = list()
    true = list()
    for index, value in enumerate(data[:-(lookback + horizon - 1)]):
        input.append(data[index:index + lookback])
        true.append(data[index + lookback:index + lookback + horizon])
    return array(input), array(true)