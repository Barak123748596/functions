import math
import numpy as np
import matplotlib.pyplot as plt

__version__ = '0.3'
__name__ = 'functions_DIY'
'''
    本模块中默认data.txt的第一行为x变量，第二行以后为y变量，切记，切记！
    若有特殊需要更改顺序，则在使用函数时定好参数！
'''



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
    arr = loadDataSet()[0]
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

#将以'\n'间隔的字符变为带'&'的LaTeX格式，这里不可用loadDataSet，因为此处有字符
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
            print(arr[i][j], '&', end=' ')
        print(arr[i][rowNum-1],'\\\\')
        print('\hline')

'''

复摆(compound pendulum)实验需要的函数
'''
#线性回归
def linear_regression(x=loadDataSet()[0], y=loadDataSet()[1]):
    #y = np.array(np.fromstring(input('Please input the y values'), sep=' '))
    #x = np.array(np.fromstring(input('Please input the x values'), sep=' '))
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
def compound_pendulum_approximate_conjugate():
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
def compound_pendulum_conjugate():
    T = float(input('T = '))
    x1 = float(input('x1 = '))/100
    x2 = float(input('x2 = '))/100
    g = 4*np.pi*np.pi*np.fabs(x1 - x2)/T/T
    print('g = ', g, 'm/s^2')

#曲线回归（局部加权线性回归Locally weighted linear regression，但是似乎不太对）
def lwlr_point(testPoint, xArr, yArr, k=0.5):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
    #    weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    print(diffMat*diffMat.T)
    print(weights)
    print(xTx)
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
def lwlr_arr(testArr,xArr,yArr,k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr_point(testArr[i],xArr,yArr,k)
    return yHat
def curve_regression(k=10):
    xArr = loadDataSet()[0]
    yArr = loadDataSet()[1]
    testPoint = 3.5  #暂时的一个测试点
    print('The predict y = ', lwlr_point(testPoint, xArr, yArr))
    #picture
    yHat = lwlr_arr(xArr, xArr, yArr, k)
    xMat = np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0] , s=2, c='red')
    plt.show()

#多项式拟合
def poly_regression(x=loadDataSet()[0], y=loadDataSet()[1], k=3):
    z1 = np.polyfit(x, y, k)#用k次多项式拟合
    p1 = np.poly1d(z1)
    print(p1) #在屏幕上打印拟合多项式
    yvals=p1(x)#也可以使用yvals=np.polyval(z1,x)
    plot1=plt.plot(x, y, '*',label='original values')
    plot2=plt.plot(x, yvals, 'r',label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
    plt.title('polyfitting')
    plt.show()
    plt.savefig('p1.png')
    testPoint = input('testPoint ? ')
    print(p1(testPoint))

#指定函数拟合
def func(a, b, x=loadDataSet()[0]):   #a,b为函数需要确定的参数
    return a*np.exp(b/x)
def specify_regression(x=loadDataSet()[0], y=loadDataSet()[1]):
    popt, pcov = np.curve_fit(func, x, y)  #没运行过，不知numpy是否有curve_fit函数
    a=popt[0]#popt里面是拟合系数，读者可以自己help其用法
    b=popt[1]
    yvals=func(x,a,b)
    plot1=plt.plot(x, y, '*',label='original values')
    plot2=plt.plot(x, yvals, 'r',label='curve_fit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)#指定legend的位置
    plt.title('curve_fit')
    plt.show()
    plt.savefig('p2.png')


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

'''介质中声速'''
def sonic_p_amend(p, t, g=9.801):
    print (round((g/9.80665*(p-(0.000182-0.00001)*p*t)),1),'mmHg')
    print (round((g/9.80665*(p-(0.000182-0.00001)*p*t)*133.3224/100000),4),"*10^5 Pa")

def sonic_v_amend(t, pw, p):
    v = 331.45*np.sqrt((1+t/273.15)*(1+0.3192*pw/p))
    print(v, 'm/s')
#计算均方根误差
def rmse():
    x = loadDataSet()[0]
    y = loadDataSet()[1]
    targets = y
    A = np.vstack([x, np.ones(len(x))]).T
    m,c = np.linalg.lstsq(A, y)[0]
    predictions = x*m+c
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    print("RMSE : ", rmse)
    return rmse
