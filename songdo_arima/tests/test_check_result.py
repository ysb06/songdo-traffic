import pickle

def test_print_result():
    with open("./output/arima-01-absolute_simple-time_mean/results.pkl", "rb") as f:
        result = pickle.load(f)
        print(result)