import numpy as np

# Test set acc

# seed 7


# Class accuracy:
# ['counting', 'math', 'logic', 'path', 'algebra', 'measure', 'spatial', 'pattern']
# 32.5 &  9.2 &  24.2 &  18.0 &  11.1 &  11.1 &  27.3 &  24.5 &


# ***** Final Test Performance: S_acc = 20.52 Prediction Variance = 0.20

# # seed 42
# https://www.comet.com/droberts308/multimodalai/62f0df93abac4f81b22923b646cf1a2c?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step

# Class accuracy:
# ['counting', 'math', 'logic', 'path', 'algebra', 'measure', 'spatial', 'pattern']
# 32.9 &  10.0 &  26.1 &  19.2 &  10.8 &  10.9 &  25.8 &  25.2 &


# ***** Final Test Performance: S_acc = 20.88 Prediction Variance = 0.20

# # seed 0

# class accuracy:
# ['counting', 'math', 'logic', 'path', 'algebra', 'measure', 'spatial', 'pattern']
# 32.9 &  10.0 &  22.8 &  19.5 &  11.2 &  11.6 &  26.3 &  25.8 &


# ***** Final Test Performance: S_acc = 20.86 Prediction Variance = 0.21


# test acc

acc = np.array([20.52, 20.88, 20.86])

print((f"overall average test accuracy {np.mean(acc)} with std dev of {np.std(acc)}"))

# test acc math skill class
acc = np.array([9.2, 10.00, 10.0])

print((f"math average test accuracy {np.mean(acc)} with std dev of {np.std(acc)}"))

# test acc pattern skill class
acc = np.array([24.5, 25.2, 25.8])

print((f"path average test accuracy {np.mean(acc)} with std dev of {np.std(acc)}"))
