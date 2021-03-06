# This Python file uses the following encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__version__ = '0.3'
__name__ = 'functions_DIY'
'''
    本模块中默认data.txt的第一行为x变量，第二行以后为y变量，切记，切记！
    若有特殊需要更改顺序，则在使用函数时定好参数！
'''
#将以'\n'间隔的字符变为带'&'的LaTeX格式，这里不可用loadDataSet，因为此处有字符
def convert_toLaTeX(hline=True):
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
        print(arr[i][rowNum-1],'\\\\', end='')
        if hline:
            for i in range(hline):
                print('\hline')
        else:   print('')


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
    if  arr.ndim == 2:  arr=np.tile(arr, (2,1))
    return arr


#积分运算
def integrate(f, a, b, N):    # N是 曲线a到b的面积分为多少个小长方形，分的越多越精确
    x = np.linspace(a, b, N)
    fx = f(x)
    area = np.integrate
    return area

#绘图
def draw_plot(x=loadDataSet()[0], y=loadDataSet()[1], title=None, xLabel=None, yLabel=None, linename=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #设置标题
    ax1.set_title(title)
    # 设置X-Y轴标签
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    #画散点图
    ax1.scatter(x, y, c='r', marker='x', label = linename)
    #设置图标
    plt.legend()
    #画平滑曲线
    from scipy.interpolate import spline
    xnew = np.linspace(x.min(), x.max(), 300) #300 represents number of points to make between x.min and x.max
    y_smooth = spline(x, y, xnew)
    plt.plot(xnew,y_smooth, 'k', linestyle = '-', marker = ',')
    #显示所画的图
    plt.show()


#标准差
def standard_deviation(tolerance=None):
    arr = loadDataSet()[0]
    print('The number of list is: ',len(arr))
    list_average = arr.mean()
    print('The average is: ', list_average)
    list_sigma = arr.std()
    print('The Standard Deviation is: ', list_sigma)
    if tolerance == None:
        print('Thank you for using, goodby ^_^')
        return
    else:
        sigma_tolerance = math.sqrt(list_sigma**2 + (tolerance**2)/3)
        print('the Standard Deviation with Tolerance is:', sigma_tolerance)
        return

#不确定度
def getSigma(x=loadDataSet()[1], getPrint=True):
    regression = linear_regression()
    r = regression.r
    u = regression.k
    n = len(x)
    sigma = u * np.sqrt( (1/r/r - 1)/(n/2) )
    if getPrint:    print('sigma = ', sigma)
    return sigma

def sigma(x=loadDataSet()[0]):
    Sum = np.sum((x-np.average(x))**2)/(np.alen(x)-1)
    sig = np.sqrt(Sum)
    print(sig)
    print(np.alen(x))


'''
复摆(compound pendulum)实验需要的函数
'''
#回归
    #线性回归
def linear_regression(x=loadDataSet()[0], y=loadDataSet()[1], newX=None, drawPlot=None, getSigma=None, getPrint = True):
        #x=loadDataSet()[0], y=loadDataSet()[1], newX=None, drawPlot=None, getSigma=None, getPrint = True
    A = np.vstack([x, np.ones(len(x))]).T
    #model
    m,c = np.linalg.lstsq(A, y)[0]
    #计算拟合优度
    SStot = ((y - y.mean())**2).sum()
    SSres = ((y - m*x - c)**2).sum()
    R2 = 1 - SSres / SStot
    r = np.sqrt(R2)
    #绘图
    if drawPlot:
            plt.plot(x, y, 'o', label = 'Original Data', markersize = 1 )
            plt.plot(x, m*x+c, 'r', label = 'Fitted line')
            plt.legend()
            plt.show()
    #predict
    if newX:
            newY = m*newX + c
            print('newY = ', newY)
            return newY
    if getSigma:
            sigma = np.sqrt((1/R2-1)/len(x))
            print("sigma = ", sigma)
            return sigma
    if getPrint:
            print('y = ', round(m, 6), 'x  + ', round(c, 6))
            print('r = ', r)
    print('Thanks for using^_^')

    #多项式拟合
def poly_regression(x=loadDataSet()[0], y=loadDataSet()[1], k=3, testPoint=None ,save=None):   #k为多项式的最高次数
        z1 = np.polyfit(x, y, k)#用k次多项式拟合
        p1 = np.poly1d(z1)
        print(p1) #在屏幕上打印拟合多项式
        yvals=p1(x)#也可以使用yvals=np.polyval(z1,x)
        plot1=plt.plot(x, y, '*',label='original values')
        plot2=plt.plot(x, yvals, 'r',label='polyfit values')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend(loc=4)#指定legend的位置
        plt.title('polyfitting')
        plt.show()
        if save:    plt.savefig('p1.png')
        if testPoint:   print("predict point : ", p1(testPoint))


#指定函数拟合
def func(a, b, x=loadDataSet()[0]):   #a,b为函数需要确定的参数
    return a*np.exp(b/x)
def specify_regression(x=loadDataSet()[0], y=loadDataSet()[1]):
    popt, pcov = np.curve_fit(func, x, y)  #没运行过，不知numpy是否有curve_fit函数
    a=popt[0]#popt里面是拟合系数
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



'''霍尔效应'''
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
def rmse(filename="data.txt"):
    x = loadDataSet(filename)[0]
    y = loadDataSet(filename)[1]
    targets = y
    A = np.vstack([x, np.ones(len(x))]).T
    m,c = np.linalg.lstsq(A, y)[0]
    predictions = x*m+c
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    print("RMSE : ", rmse)
    return rmse

'''迈克尔逊干涉仪'''
#计算气压折射率
def n0_sigma(lambda_prime=632.8, D=4.00):
    a=linear_regression()
    k = a.k
    n0 = 1 + lambda_prime/2/D/k/10000
    sigma = getSigma(getPrint=False)
    print('n0 ± sigma = ', n0, '±', sigma)

'''杨氏模量'''
class Young_modulus():
    def __init__(self):
        self.g = 9.80
        self.k =0.00059
        self.L = 775
        self.d = 0.328
        self.sigma_d = 0.006
    def E(self):
        E = 4 * self.g * self.L / (np.pi * self.d * self.d * self.k) * 1000
        print(np.round(E/10**11, 2), '×10^11 Pa')
        return E

'''分光计'''
def deg2rad(x = loadDataSet()):
    y = np.round(x, 0)  + (np.round(x, 2)-np.round(x, 0))/60 + (np.round(x, 4)-np.round(x, 2))/360
    z = y * 2 * np.pi / 360
    print(z[0])
