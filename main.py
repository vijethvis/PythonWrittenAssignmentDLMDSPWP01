# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pandas
import sqlite3 as sql3
from sqlalchemy import create_engine
from scipy import optimize
from bokeh.plotting import figure, output_file, show
import logging
import sys

LOG = logging.getLogger(__name__)

def readthecsv(fName):
    try:
        cdata = pandas.read_csv(fName)
    except FileNotFoundError:
        LOG.info("{} file cannot be found".format(fName))
        raise
    return cdata

def func(x, a, b):
    y = a * x + b
    return y

def getxcol(infile1):
    return infile1.iloc[:, 0]

def optimisecurve(xcol, ycol):
    return optimize.curve_fit(func, xdata=xcol, ydata=ycol)[0]

def calc_error_sqr(lnfile, xcol):
    iii = lnfile.iloc[:, :].shape[1]
    jjj = lnfile.iloc[:, :].shape[0]
    a = []
    for j in range(iii - 1):
        if j > 51:
            continue
        ycol = lnfile.iloc[:, j]
        #  sumx = xcol.sum(0)
        #  sumy1 = ycol.sum(0)
        #  sqsumx = xcol.pow(2).sum()
        #  sumxy1 = (xcol * ycol).sum()
        newlistofy = repeatfunc(xcol, ycol)
        delta_y = ((newlistofy - ycol) ** 2).sum() / (jjj)
        a.append((delta_y, j))
    LOG.info(a)
    LOG.info(min(a))
    return min(a)


def repeatfunc(xcol, ycol):
    alpha = optimisecurve(xcol, ycol)
    aa = alpha[1]
    bb = alpha[0]
    newlistofy = ycol - (aa + (bb * xcol))
    return newlistofy


def plotgraph(minval, xcol, lfile1):
    indexy = minval[1]
    ycol = lfile1.iloc[:, indexy]
    newlistofy = repeatfunc(xcol, ycol)
    output_file("gfg.html")
    graph = figure(title="Bokeh Line Graph")
    graph.circle_dot(xcol, newlistofy, size=5, color="red", alpha=0.5)
    graph.line(xcol, newlistofy, color="grey")
    show(graph)

def loadtosql(fname, lstring):
    try:
        dbEngine = create_engine('sqlite:///information.db', echo=False)
        fname.to_sql(lstring, dbEngine, if_exists='replace', index=False)
    except sql3.Error as error:
        LOG.info("Failed to load data to table", error)
    # finally:
    #     if sql3.connect('information.db'):
    #         sql3.connect('information.db').close()
    #     LOG.info(sql3.connect('information.db'))

def docomputation(lfilen):
    xcol = getxcol(lfilen)
    return calc_error_sqr(lfilen, xcol)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    'Step 1 : Input Data declaration'
    idealCsvPath = "C:/Users/Manzviju/Desktop/Project_Python/DataCSV/ideal.csv"
    trainCsvPath = "C:/Users/Manzviju/Desktop/Project_Python/DataCSV/train.csv"
    testCsvPath = "C:/Users/Manzviju/Desktop/Project_Python/DataCSV/test.csv"


    logging.basicConfig(
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )


    'Step 2 : Reading the input file and load to sql'
    testcsv = readthecsv(testCsvPath)
    traincsv = readthecsv(trainCsvPath)
    idealcsv = readthecsv(idealCsvPath)

    'Step 3 : move csv to sqlite3'
    # loadtosql(testcsv, "testdata")
    loadtosql(traincsv, "traindata")
    loadtosql(idealcsv, "idealdata")

    'Step 4 : Convert data frames to array '
    minvalf = docomputation(idealcsv)
    xcol = getxcol(idealcsv)
    plotgraph(minvalf, xcol, idealcsv)

