import functions_DIY
import numpy as np
x = functions_DIY.loadDataSet()[0]
y = functions_DIY.loadDataSet()[1]
functions_DIY.lwlr_point(3.5, x, y, k = 1)
#functions_DIY.curve_regression(k=1)
