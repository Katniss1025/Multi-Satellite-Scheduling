import random


def try_sample(X, dist):
    #prob = random.random()
    x = random.random()*3.5*1e6
    p = dist.predict(x)['y_proba']
    if p >= random.random(): #轮盘赌
        X.append(x)
    return X


def sample(length, dist):
    X = []
    while len(X) != length:
        X = try_sample(X, dist)
    return X


if __name__ == '__main__':
    from distfit import distfit
    dist = distfit()
    dist.load('interval_generator.pkl')
    X = sample(10, dist)
    print(X)