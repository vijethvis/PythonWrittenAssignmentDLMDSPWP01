'''
Assumptions:
1. The input folder is present and the path is set.
2. file names should be ideal.csv, test.csv and train.csv within input folder.
3. train.csv and ideal.csv should have equal number of rows and X column should be same.
'''

'''import pandas, mean_squared_error and LinearRegression, pysqlite3, sqlalchemy, numpy, sklearn, scikit-learn, logging, sys, bokeh'''
import pandas as pd
import os
import logging
import sys
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show
import sqlite3

LOG = logging.getLogger(__name__)
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

'''Read the files from input folder named input and place ideal.csv, test.csv and train.csv inside the folder'''
try:
    train_df = pd.read_csv(r"input/train.csv")
    test_df = pd.read_csv(r"input/test.csv")
    ideal_df = pd.read_csv(r"input/ideal.csv")
except FileNotFoundError:
    print("One or more input files not found. Check the path variable and filename")
    exit()
LOG.info('s1: Read input files')

class loadtosqlite3(object):
    def __init__(self,ilist,lstring):
        dbEngine = create_engine('sqlite:///information.db', echo=False)
        ilist.to_sql(lstring, dbEngine, if_exists='replace', index=False)

class getsorteddata(loadtosqlite3):
    def __init__(self, ldf, lstring):
        self.ldf = ldf
        self.lstring = lstring
        self.ilist = self.ldf.sort_values(by=["x"])
    def sorteddataf(self):
        loadtosqlite3.__init__(self, self.ilist, self.lstring)
        return self.ilist

def plotgraph(x, y, ychoose, ytrainlabel, ychooselabel):
    graph = figure(title="Best chosen function vs train function", x_axis_label='x', y_axis_label="TrainY: " +
                ytrainlabel + "vs" + "BestY: " + ychooselabel)
    graph.circle(x, y, color="red", legend_label=ytrainlabel, size=2.5)
    graph.line(x, ychoose, color="grey", legend_label=ychooselabel, line_width=1.5)
    graph.legend.location = "top_left"
    show(graph)

def getallycols(ldataf):
    return [x for x in list(ldataf.columns.values) if x != "x"]

'''sort the data frames based on X column for train and ideal csv and load to sqlite3 db'''
g = getsorteddata(train_df, "train")
train_sorted = g.sorteddataf()
LOG.info('s2: Sorted train data and loaded to sqlitedb table name traindata')

f = getsorteddata(ideal_df, "ideal")
ideal_sorted = f.sorteddataf()
LOG.info('s3: Sorted ideal data and loaded to sqlitedb table name idealdata')

''' getting the column names apart from X which are Y columns from the dataframes'''
train_cols = getallycols(train_df)
ideal_cols = getallycols(ideal_df)
LOG.info('s4: got train and ideal columns')

'''initialising the variables ideal_func as dictionary and result_df as pandas dataframes
assign X column from train_sorted to result_df'''
ideal_func = {}
result_df = pd.DataFrame()
result_df["x"] = train_sorted["x"]
LOG.info('s5: initialising result_df by loading X')

'''Looping through each train and ideal data of y to find all possible mean square error and fetching best four'''
for i in train_cols:
    mse_list = []
    result_df[i] = train_sorted[i]
    for j in ideal_cols:
        mse = mean_squared_error(train_sorted[i], ideal_sorted[j])
        mse_list.append((i, j, mse))
    mse_list.sort(key=lambda a: a[2])
    best_mse = mse_list[0]
    ideal_func[i] = best_mse[1]
    col_name = str(i)+"_"+str(best_mse[1])
    result_df[col_name] = ideal_sorted[best_mse[1]]
LOG.info('s6: looped through 400 possible combinations of ideal and train datasets and got resultdf')
# print(result_df)
'''Plot four graphs for train and chosen best fit ideal functions'''
for k in range(1, 8, 2):
    lvar = "IdealvsTrain" + str(k) + ".html"
    if os.path.exists(lvar):
        os.remove(lvar)
    output_file(lvar)
    colst = list(result_df.columns)
    plotgraph(result_df.iloc[:, 0], result_df.iloc[:, k], result_df.iloc[:, k+1], colst[k], colst[k+1])

'''Calculating coefficients of Linear regression model for best 4 functions selected'''
model_coef = {}
for k in ideal_func.keys():
    x = ideal_df["x"].values.reshape(-1, 1)
    y = ideal_df[ideal_func[k]]
    reg = LinearRegression(fit_intercept=False).fit(x, y)
    model_coef[ideal_func[k]] = reg.coef_
LOG.info('s7: Linear regression done and saved co efficients')
# print(model_coef)
'''Calculating delta values of y test data with selected functions '''
for k in model_coef.keys():
    test_df[k] = test_df["x"]*model_coef[k]
    test_df[k+"_delta"] = test_df[k] - test_df["y"]
    test_df[k + "_delta"] = test_df[k+"_delta"].abs()
col_name_list = [x for x in test_df.columns if x.endswith("_delta")]
test_df["delta"] = test_df[col_name_list].min(axis=1)
LOG.info('s8: Calculated delta values of y test data with selected four functions')

'''finding ideal column for each and every row of test data'''
idealcolumn = []
for index, row in test_df.iterrows():
    for col in col_name_list:
        if row[col] == row["delta"]:
            idealcolumn.append(col.split("_")[0])
            break
LOG.info('s9: found ideal column for each and every row of test data')

'''adding ideal func column to the dataset'''
test_df["ideal_func"] = pd.Series(idealcolumn)
# print(test_df.columns)
'''getting only required columns to final_df which are x,y,delta and ideal function number'''
final_df = test_df[["x", "y", "delta", "ideal_func"]]
LOG.info('s10: got final_df which are x,y,delta and ideal function number')

'''loaded final_df to sqlite3 db'''
loadtosqlite3(final_df, "test_data_ydev")
LOG.info('s11: loaded test_data_ydev table to sqlite3')

opfile = "Error_Report.html"
if os.path.exists(opfile):
    os.remove(opfile)
output_file(opfile)
finalgraph = figure(title="Error_Report",
                    x_axis_label='x', y_axis_label="Y actual and Delta Y")
finalgraph.circle(final_df.iloc[:, 0], final_df.iloc[:, 1], color="red", legend_label="y Test", size=2.5)
finalgraph.circle(final_df.iloc[:, 0], final_df.iloc[:, 2], color="grey", legend_label="y Delta", size=2.5)
finalgraph.legend.location = "top_left"
show(finalgraph)

conn = sqlite3.connect('information.db', isolation_level=None,
                       detect_types=sqlite3.PARSE_COLNAMES)
db_df = pd.read_sql_query("SELECT * FROM test_data_ydev", conn)
db_df.to_csv('Error_Report.csv', index=False)
