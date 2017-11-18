import math
import numpy as np
import matplotlib.pyplot as plt

#绘图
def draw_plot(x, y):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #设置标题
    ax1.set_title(input('title : '))
    # 设置X-Y轴标签
    plt.xlabel(input('xLabel : '))
    plt.ylabel(input('yLabel : '))
    #画散点图
    ax1.scatter(x, y, c='r', marker='x', label = input('Line name : '))
    #设置图标
    plt.legend()
    #画平滑曲线
    from scipy.interpolate import spline
    xnew = np.linspace(x.min(), x.max(), 300) #300 represents number of points to make between x.min and x.max
    y_smooth = spline(x, y, xnew)
    plt.plot(xnew,y_smooth, 'k', linestyle = '-', marker = ',')
    #显示所画的图
    plt.show()


#得到点列
def loadDataSet(filename="data.txt"):
    file = open(filename,"r")
    list_arr = file.readlines()
    lists = []
    for index,x in enumerate(list_arr):
        x = x.strip()
        x = x.strip('\n')
        x = x.split('\t')
        lists.append(x)
    file.close()
    arr = np.array(lists).astype(float)
    return arr

#标准差
def standard_deviation():
    arr = np.fromstring(input('input the list:\n'), dtype=float, sep=' ')
    print('The number of list is: ',len(arr))
    list_average = arr.mean()
    print('The average is: ', list_average)
    list_sigma = arr.std()
    print('The Standard Deviation is: ', list_sigma)
    print()
    judge = input('Do you want to get the standard deviation adding tolerance(y or n)?\n')
    if judge == 'n':
        print('Thank you for using, goodby ^_^')
        return
    else:
        tolerance = float(input('input the Tolerance:'))
        sigma_tolerance = math.sqrt(list_sigma**2 + (tolerance**2)/3)
        print('the Standard Deviation with Tolerance is:', sigma_tolerance)
        return

#将以'\n'间隔的字符变为带'&'的LaTeX格式

def convert_toLaTeX():
    file = open("data.txt","r")
    list_arr = file.readlines()
    lists = []
    for index,x in enumerate(list_arr):
        x = x.strip()
        x = x.strip('\n')
        x = x.split('\t')
        lists.append(x)
    arr = np.array(lists)
    file.close()
    lineNum = arr.shape[0]
    rowNum = arr.shape[1]
    for i in range(lineNum):
        for j in range(rowNum-1):
            print(arr[i][j], ' & ', end=' ')
        print(arr[i][rowNum-1],'\\\\')

'''

复摆(compound pendulum)实验需要的函数
'''
#线性回归
def linear_regression():
    y = np.array(np.fromstring(input('Please input the y values'), sep=' '))
    x = np.array(np.fromstring(input('Please input the x values'), sep=' '))
    A = np.vstack([x, np.ones(len(x))]).T
    #model
    m,c = np.linalg.lstsq(A, y)[0]
    print('y = ', round(m, 6), 'x  + ', round(c, 6))
    #计算拟合优度
    SStot = ((y - y.mean())**2).sum()
    SSres = ((y - m*x - c)**2).sum()
    R2 = 1 - SSres / SStot
    print('The coefficient of determination is', R2)
    #绘图
    plt.plot(x, y, 'o', label = 'Original Data', markersize = 1 )
    plt.plot(x, m*x+c, 'r', label = 'Fitted line')
    plt.legend()
    plt.show()
    #predict
    if(input('Do you want to predict(y or n)?') == 'y'):
        newX = float(input('newX = '))
        print('newY = ',m*newX + c)
    else:print('Thanks for using^_^');

#近似共轭点法
def compound_pendulum_jinsigongedian():
    h1 = float(input('h1 = '))/100
    h2 = float(input('h2 = '))/100
    T1 = float(input('T1 = '))
    T2 = float(input('T2 = '))
    A = (T1*T1 + T2*T2)/2/(h1 + h2)
    B = (T1*T1 - T2*T2)/2/(h1 - h2)
    g = 8*np.pi*np.pi*(h1 + h2)/(T1*T1 + T2*T2)
    print('A = ', round(A, 2) , 'm^-1s^2')
    print('B = ', round(B, 3) , 'm^-1s^2')
    print('g = ', round(g, 2), 'm/s^2')

#共轭点法
def compound_pendulum_gongedian():
    T = float(input('T = '))
    x1 = float(input('x1 = '))/100
    x2 = float(input('x2 = '))/100
    g = 4*np.pi*np.pi*np.fabs(x1 - x2)/T/T
    print('g = ', g, 'm/s^2')

#曲线回归（局部加权线性回归Locally weighted linear regression）
def lwlr_point(testPoint,xArr,yArr,k=0.01):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    print(m)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        print('diffmat = ',diffMat)
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    print(xTx)
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
def lwlr_arr(testArr,xArr,yArr,k=0.01):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr_point(testArr[i],xArr,yArr,k)
    return yHat
def curve_regression():
    xArr = np.fromstring(input('xArr = '), sep=' ')
    yArr = np.fromstring(input('yArr = '), sep=' ')
    testPoint = float(input('testPoint x = '))
    print('The predict y = ', lwlr_point(testPoint, xArr, yArr))
    #picture
    yHat = lwlr_arr(xArr, xArr, yArr,0.003)
    xMat=np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0] , s=2, c='red')
    plt.show()

'''

霍尔效应
'''
def from_I_U_K_to_B():
    I_M = float(input('input I_H: '))
    U_H = np.fromstring(input('input U_H: '), sep='\t')
    K_H = float(input('K_H: '))
    B = U_H/K_H*1000/I_M
    print('B: ')
    for b in B:
        print(round(b,1), end='\t')
    print('mT')

'''光学折射角'''
def compute_delta(a, n1, n2=1):
    b = a/180*np.pi
    delta_p = np.array([])
    delta_s = np.array([])
    print("delta_p    delta_s")
    for i in range(len(a)):
        delta_p = np.append(delta_p, 2*np.arctan(n1/n2/np.cos(b[i])*np.sqrt((n1/n2*np.sin(b[i]))*(n1/n2*np.sin(b[i]))-1)))
        delta_s = np.append(delta_s, 2*np.arctan(n2/n1/np.cos(b[i])*np.sqrt((n1/n2*np.sin(b[i]))*(n1/n2*np.sin(b[i]))-1)))
        delta_p[i] = int(delta_p[i]/np.pi*180)
        delta_s[i] = int(delta_s[i]/np.pi*180)
        print(a[i], delta_p[i], delta_s[i])
    #画平滑曲线
    from scipy.interpolate import spline
    aNew = np.linspace(a.min(), a.max(), 3000) #300 represents number of points to make between x.min and x.max
    delta_p_smooth = spline(a, delta_p, aNew)
    delta_s_smooth = spline(a, delta_s, aNew)
    #显示所画的图
    plt.plot(aNew, delta_p_smooth, 'o', label = 'delta_p', linestyle = '-', marker = ',')
    plt.plot(aNew, delta_s_smooth, 'r', label = 'delta_s', linestyle = '-', marker = ',')
    plt.show()
