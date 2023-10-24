
from doctest import DocFileSuite
from turtle import color
from BuildModel import ModelBuild
import pandas as pd
import os
import seaborn as sns
# https://seaborn.pydata.org/
import matplotlib.pyplot as plt
# https://matplotlib.org/stable/

#Import for step 6
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from bokeh.plotting import figure, output_notebook, show
from bokeh.palettes import Turbo256
from bokeh.transform import linear_cmap

from matplotlib import pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# help
# https://github.com/AxelThevenot/Python_For_Data_Analysis_Seoul_Bike
# https://github.com/Pierre-Portfolio/Seoul-Bike-Sharing/blob/main/livrable/jupyter/SeoulBike.ipynb

import numpy as np

def Bike_Data():

    # Get Data from:
    # https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand
    
    #Name of the data file
    filename = "SeoulBikeData.csv"
    #File location - uses work directory and specific folder 
    filePath = os.getcwd() + "\inputFiles\{}".format(filename)
    # Encoding fuction - Finds the encoding that are used for the data file
    Encoding = ModelBuild.getEncoding(filename)["encoding"]
    #Reads in the data in a pandas dataframe
    bikeData = pd.read_csv(filePath, encoding=Encoding)
    # Date column to datetime
    bikeData["Date"] = pd.to_datetime(bikeData["Date"], format="%d%m%Y", infer_datetime_format=True, dayfirst=True)
    # Column with years
    bikeData["Year"] = bikeData["Date"].dt.year
    # Column with Months
    bikeData["Month"] = bikeData["Date"].dt.month_name().str[:3]
    # Column with Weekday
    bikeData["Weekday"] = bikeData["Date"].dt.weekday
    # Check for nan values = There is none
    # print(bikeData.isnull().sum())
    return bikeData



def Category_YesNo():
    return {"Yes": '1', "No": '0'}


# Function for computing time for the scoring fitting and returning a tuple
def TimeOfModel(func,*args):
    start_time = time.time()
    return func(*args) , round(time.time() - start_time,3)

# For each model we display the r2 accuracy and we only store the best value
def BestModelWithoutTraining(namefunc, func, listFeatures, Y_Series):
    y = Y_Series
    bestModel = 1
    bestModelTime = 0
    bestModelExactValue = 0
    incModel = 1
    
    for x in listFeatures:
        exactmodel = TimeOfModel(func, x, y)
        print(namefunc + 'Model ' + str(incModel) + ' r2 accuracy : ' + str(exactmodel[0]) + '% in ' + str(exactmodel[1]) + 'secondes')
        if(bestModelExactValue < exactmodel[0]):
                bestModel = incModel
                bestModelTime = exactmodel[1]
                bestModelExactValue = exactmodel[0]
        incModel = incModel + 1
    print("Best exact" + namefunc + " model is : " + str(bestModel) + " with " + str(bestModelExactValue) + '%')
    return [namefunc, str(bestModel), bestModelTime, bestModelExactValue]


# Function for taking the gridsearch on all models
def BestModelWithTraining(namefunc, func, algo, hyperparametres, listFeatures, Y_Series):
    bestModel = 1
    bestModelTime = 0
    bestModelExactValue = 0
    incModel = 1
    for i in listFeatures:
        print('Computing model ' + str(incModel) + ' in progress ...')
        x_train, x_test, y_train, y_test = train_test_split(i, Y_Series, test_size=0.33)
        
        #Scale
        scaler = StandardScaler()
        scaler.fit(x_train)       
        x_train = scaler.transform(x_train, copy = False)
        x_test  = scaler.transform(x_test, copy = False)
        
        #Find Model
        InfoModel , time = TimeOfModel(func, algo, x_train, y_train, hyperparametres)           
        
        #Save the best model
        if(bestModelExactValue < InfoModel[1]):
                bestModel = incModel
                bestModelTime = time
                bestModelExactValue = InfoModel[1]
        incModel = incModel + 1
    print("Best r2 accuracy " + namefunc + " model is : " + str(bestModel) + " with " + str(bestModelExactValue) + '%')
    return [namefunc, bestModel, bestModelTime, bestModelExactValue]



# Generate a dataframe or store the best model for each method
ResultDf = pd.DataFrame()
def StockResultDf(tab):
    global ResultDf
    df2 = {'Model Name': tab[0], 'Model Number': tab[1], 'Time(s)': tab[2], 'R2 Value (%)' : tab[3]}
    ResultDf = ResultDf.append(df2,ignore_index=True)


def KNN(x,y):
    KnnModel = KNeighborsClassifier(n_neighbors=2)
    KnnModel.fit(x,y)
    return round(KnnModel.score(x,y) * 100,2)


# Linear Regression Model
def LinearRegressionModel(x,y):
    LinearModel = LinearRegression()
    LinearModel.fit(x, y)
    return round(LinearModel.score(x,y) * 100,2)


# Function which returns the grid fitting
def GetScoreHyperparametres(algo, x, y, hyperparametres):
    grid = GridSearchCV(algo, hyperparametres, n_jobs=-1)
    grid.fit(x, y)
    ExactScore = round(grid.score(x,y) * 100,2)
    
    print('Best r2 accuracy for this model : ' + str(ExactScore) + '% with hyperparameters : ' + str(grid.best_estimator_))
    return grid , ExactScore

def xgBoost_test():
        # Read in the Bike rental data
    df = Bike_Data()
    is_nan = df.isna().sum() 

    # Replace " " with "_" in all columns
    df.columns = [s.strip().replace(" ", "_") for s in df.columns] # all columns

    # Loop over the dataframe columns. If the dtype is == object, then append to list
    columns_object_list = []
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        if dtype == "object":
            columns_object_list.append(col)
    
    # For columns in columns_object_list. 
    # Get unique values. Enumerate over unique values and create new categories
    for col in columns_object_list:
        unique_values = df[col].unique()
        unique_category = {}
        for idx, val in enumerate(unique_values):
            unique_category[val] = idx + 1
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.rename_categories(unique_category)
    
    # Rename Weekday and Turn into categories
    df.Weekday = df.Weekday.map({
        0: "Monday", 
        1: "Tuesday",
        2: "Wednesday" ,
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
        })

    df["Weekday"] = pd.Categorical( 
        df["Weekday"], 
        categories = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
        ordered=True
        )

    # Rename Seasons and Turn into categories
    df.Seasons = df.Seasons.map({
        1: "Winter",
        2: "Spring",
        3: "Summer",
        4: "Autumn"
    })

    df["Seasons"] = pd.Categorical(
        df["Seasons"],
        categories = ['Winter','Spring','Summer','Autumn'],
        ordered=True
    )
    
    #Rename Month to jan, feb... ect
    df["Month"] = df["Date"].dt.month_name().str[:3]

def main():
    '''
    Step 1:     Read in data - Bike Rental
    Step 2:     Replace " " with "_" in all columns
    Step 3:     Turn all object columns to Category
    Step 4:     Rename Categories to 1, 2, 3... ect
    Step 5:     Turn Season, Month & Weekday into Categories


    Step 5:     Identify Independent & dependend variables
    Step 6:     Inisialise ModelBuild
                Train_size = 60%, Test_Size = 40%
    Step 7:     Use Exhaustive Search to find best variable combination

    '''
    
    # Read in the Bike rental data
    df = Bike_Data()
    is_nan = df.isna().sum() 

    # Replace " " with "_" in all columns
    df.columns = [s.strip().replace(" ", "_") for s in df.columns] # all columns

    # Loop over the dataframe columns. If the dtype is == object, then append to list
    columns_object_list = []
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        if dtype == "object":
            columns_object_list.append(col)
    
    # For columns in columns_object_list. 
    # Get unique values. Enumerate over unique values and create new categories
    for col in columns_object_list:
        unique_values = df[col].unique()
        unique_category = {}
        for idx, val in enumerate(unique_values):
            unique_category[val] = idx + 1
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.rename_categories(unique_category)
    
    # Rename Weekday and Turn into categories
    df.Weekday = df.Weekday.map({
        0: "Monday", 
        1: "Tuesday",
        2: "Wednesday" ,
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
        })

    df["Weekday"] = pd.Categorical( 
        df["Weekday"], 
        categories = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
        ordered=True
        )

    # Rename Seasons and Turn into categories
    df.Seasons = df.Seasons.map({
        1: "Winter",
        2: "Spring",
        3: "Summer",
        4: "Autumn"
    })

    df["Seasons"] = pd.Categorical(
        df["Seasons"],
        categories = ['Winter','Spring','Summer','Autumn'],
        ordered=True
    )
    
    #Rename Month to jan, feb... ect
    df["Month"] = df["Date"].dt.month_name().str[:3]
    

    # Data information

    print(df.shape)
    print(df["Date"].iloc[0], df["Date"].iloc[-1])


    #######################################################################################################
    # Time Series - Scatterplot 
    #######################################################################################################
    # date_series = pd.to_datetime(df["Date"])
    # y_series = df["Rented_Bike_Count"]
    # plt.style.use('dark_background')
    # fix, ax1 = plt.subplots(figsize=(16,8), )
    # sns.scatterplot(x=date_series, y=y_series, ax = ax1)
    # ax1.set_title("Time Series - Scatterplot of Rented Bike Count")
    # ax1.set_xlabel("Date")
    # ax1.set_ylabel("Rented Bike Count")
    # plt.show()
    #######################################################################################################


    #######################################################################################################
    # Time Series - Line plot - Avarage week 
    #######################################################################################################
    # df["Week"] = df["Date"].dt.week
    # df_noZero = df.loc[df["Rented_Bike_Count"] > 0]
    # grp_week = df_noZero.groupby("Week").mean().reset_index()
    # plt.style.use('dark_background')
    # fix, (ax1, ax2) = plt.subplots(2, figsize=(12,6), gridspec_kw={'hspace': 0.5})

    # sns.lineplot(data=df_noZero, x="Date", y="Rented_Bike_Count", ax=ax1)
    # ax1.set_title("Time Series - Line plot of Rented Bike Count Date")
    # ax1.set_xlabel("Date")
    # ax1.set_ylabel("Rented Bike Count")

    # sns.lineplot(data=grp_week, x="Week", y="Rented_Bike_Count", ax=ax2)
    # ax2.set_title("Time Series - kde plot of Rented Bike Count Week average")
    # ax2.set_xlabel("Week")
    # ax2.set_ylabel("Rented Bike Count")
    
    # plt.show()
    #######################################################################################################


    #######################################################################################################
    # Groupby Month - Sum - Bar Chart
    #######################################################################################################
    # grp_month_mean = df.groupby("Month").sum().reset_index()
    # plt.style.use('dark_background')
    # fig ,ax1 = plt.subplots(figsize=(16,8), gridspec_kw={'hspace': 0.3})
    # MonthsOrder = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # sns.barplot(data=grp_month_mean,x="Month",y="Rented_Bike_Count",palette="colorblind",order = MonthsOrder,ax = ax1,errorbar=None)
    # ax1.set_title("Month Bike Rents Sum")
    # ax1.set_xlabel("Month")
    # plt.show()
    #######################################################################################################

    #######################################################################################################
    # # Groupby Month - Sum - BoxPlot
    #######################################################################################################    
    # plt.style.use('dark_background')
    # MonthsOrder = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # fig, ax = plt.subplots()
    # fig.set_size_inches((12,4))
    # sns.boxplot(x='Month', y="Rented_Bike_Count", data=df ,ax=ax, order = MonthsOrder)
    # plt.show()
    #######################################################################################################

    #######################################################################################################
    # #Seasonality and Cycle illustration
    #######################################################################################################
    # plt.style.use('dark_background')
    # fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,8), gridspec_kw={'hspace': 0.3})
    # sns.barplot(data=df,x="Seasons",y="Rented_Bike_Count",palette="colorblind",ax=ax1,errorbar=None)
    # sns.lineplot(data=df,x="Month",y="Rented_Bike_Count",color="tab:blue",ax=ax3,errorbar=None)
    # sns.barplot(data=df,x="Weekday",y="Rented_Bike_Count",palette="colorblind",ax=ax2,errorbar=None)
    # sns.lineplot(data=df,x="Hour",y="Rented_Bike_Count",color="tab:blue",ax=ax4,errorbar=None)
    # ax1.set_title("Seasonal Bike Rents")
    # ax3.set_title("Monthly Bike Rents")
    # ax1.set_xlabel("Season")
    # ax3.set_xlabel("Month")
    # ax2.set_title("Weekday Bike Rents")
    # ax4.set_title("Hourly Bike Rents")
    # ax2.set_xlabel("")
    # plt.savefig("Timeline.png",dpi=300)
    # plt.show()
    #######################################################################################################
    
    #######################################################################################################
    # Weekday hour pointplot
    #######################################################################################################
    # plt.style.use('dark_background')
    # sns.pointplot(x = "Hour", y = "Rented_Bike_Count", hue = "Weekday", data=df)
    # plt.title("Rented Bikes by weekday and hour", fontsize=15)
    # plt.xlabel('Hour', fontsize=16)
    # plt.ylabel('Rented Bike Count', fontsize=16)
    # plt.show()
    #######################################################################################################

    #######################################################################################################
    # Repartition of bikes rental by Season - Pie
    #######################################################################################################
    # Winter=df.loc[df["Seasons"] == "Winter"].sum()
    # Spring=df.loc[df["Seasons"] == "Spring"].sum()
    # Summer=df.loc[df["Seasons"] == 'Summer'].sum()
    # Autumn=df.loc[df["Seasons"] == 'Autumn'].sum()

    # BikeSeasons={
    #     "Winter": Winter["Rented_Bike_Count"], 
    #     "Spring": Spring["Rented_Bike_Count"],
    #     "Summer": Summer["Rented_Bike_Count"],
    #     "Autumn": Autumn["Rented_Bike_Count"]
    #     }
    # plt.style.use('dark_background')
    # plt.style.use('_mpl-gallery-nogrid')
    # colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(BikeSeasons.values())))
    
    # plt.gcf().set_size_inches(16,8)
    # plt.pie(BikeSeasons.values(),labels = BikeSeasons.keys(), colors=colors, autopct='%1d%%', textprops={'color':"white"})
    # plt.title("Repartition of bikes rental by Season", fontsize=30, color="white")
    # plt.show()
    #######################################################################################################

    #######################################################################################################
    # Bike Rental workday cs weekend - by houers
    #######################################################################################################
    # df["Weekday_Number"] = df["Date"].dt.weekday
    # df['Weekday_Number'] += 1 
    # BikeHoursWorkDay={}
    # for k in range(24):
    #     BikeHoursWorkDay[k] = df[(df["Hour"] == k) & df["Weekday_Number"].between(1, 5)].sum()["Rented_Bike_Count"]

    # BikeHoursWeekEnd={}
    # for k in range(24):
    #     BikeHoursWeekEnd[k] = df[(df["Hour"] == k) & df["Weekday_Number"].between(6, 7)].sum()["Rented_Bike_Count"]
    
    # plt.style.use('dark_background')
    # fig, ax = plt.subplots()
    # p1 = ax.bar(BikeHoursWorkDay.keys(), BikeHoursWorkDay.values(), color='b', label ="WorkDay")
    # p2 = ax.bar(BikeHoursWeekEnd.keys(), BikeHoursWeekEnd.values(), color='r', label ="WeekEnd", width= 0.4)
    # plt.gcf().set_size_inches(15,8)
    # plt.title("Repartition of bikes rental by hours", fontsize = 25)
    # plt.xlabel('Hours', fontsize = 16)
    # plt.ylabel('Rented Bike Count', fontsize = 16)
    # fig.tight_layout()
    # ax.legend(fontsize=20)
    # plt.show()
    #######################################################################################################

    #######################################################################################################
    # Correlation - Full
    #######################################################################################################
    # corr2 = df.corr()
    # #Plot figsize and Generate Color Map
    # plt.style.use('dark_background')
    # fig, ax = plt.subplots(figsize=(20, 8))
    # colormap = sns.diverging_palette(220, 10, as_cmap=True)

    # #Generate Heat Map, allow annotations and place floating values in map
    # sns.heatmap(corr2, cmap="Blues", annot=True, fmt=".2f").set_title("Correlate the features",fontsize = 20)

    # #Apply ticks
    # plt.xticks(range(len(corr2.columns)), corr2.columns);
    # plt.yticks(range(len(corr2.columns)), corr2.columns)

    # plt.show()
    #######################################################################################################

    #######################################################################################################
    #Correlation - On Weather parameters
    #######################################################################################################
    # corr = df[
    #     ["Rented_Bike_Count", 'Temperature(°C)', 'Humidity(%)',
    #    'Wind_speed_(m/s)', 'Visibility_(10m)', 'Dew_point_temperature(°C)',
    #    'Solar_Radiation_(MJ/m2)', 'Rainfall(mm)', 'Snowfall_(cm)']
    # ].corr()
    #######################################################################################################

    #######################################################################################################
    # Temperature rented bikes count
    #######################################################################################################
    # temp_as_int = df["Temperature(°C)"].astype(int)
    # plt.style.use('dark_background')
    # fig,(ax1) = plt.subplots(figsize=(16,8), gridspec_kw={'hspace': 0.3})
    # sns.barplot(data=df, x = temp_as_int, y = "Rented_Bike_Count", palette = "colorblind", ax = ax1, errorbar = None)
    # plt.show()
    #######################################################################################################

    #######################################################################################################
    # Heatmap
    #######################################################################################################
    # # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # plt.style.use('dark_background')
    # sns.heatmap(data=corr[["Rented_Bike_Count"]].sort_values(by='Rented_Bike_Count', ascending=False), vmin=-1, vmax=1,center=0,cmap="coolwarm",annot=True,square=True,linewidths=.5)
    # plt.title("Correlation with weather parameters")
    # # plt.savefig("Correlation.png",dpi=300)7
    # plt.show()
    #######################################################################################################


    #######################################################################################################
    # Rented Bike sum pr month & Mean Temperature pr month
    #######################################################################################################
    # bike_rental_month_sum = df.groupby("Month").sum().reset_index()
    # temp_month_mean = df.groupby("Month").mean().reset_index()
    # bike_rental_and_temp_Frame = pd.DataFrame({"Month": bike_rental_month_sum["Month"], "Rented_Bike_Count": bike_rental_month_sum["Rented_Bike_Count"], "Temperature(°C)": temp_month_mean["Temperature(°C)"]})
    # bike_rental_and_temp_Frame["month_sort"]= pd.to_datetime(bike_rental_and_temp_Frame.Month, format='%b', errors='coerce').dt.month
    # bike_rental_and_temp_Frame = bike_rental_and_temp_Frame.sort_values(by="month_sort").reset_index(drop=True)
    # plt.style.use('dark_background')
    # fig, ax1 = plt.subplots(figsize=(12,6))
    # sns.barplot(data = bike_rental_and_temp_Frame, x="Month", y="Rented_Bike_Count", alpha=0.5, ax=ax1)
    # ax2 = ax1.twinx()
    # sns.lineplot(data = bike_rental_and_temp_Frame, x="Month", y="Temperature(°C)", marker='o', sort = False, ax=ax2)   
    # plt.show()
    #######################################################################################################

    #######################################################################################################

    #######################################################################################################
    # # We group by Month and Hour for scattler plotting
    # df2 = Bike_Data()
    # # Replace " " with "_" in all columns
    # df2.columns = [s.strip().replace(" ", "_") for s in df.columns] # all columns

    # print(df2.columns)


    # dfMonth = df2.groupby('Month').agg({
    #     'Rented_Bike_Count': ['sum'], 
    #     'Temperature(°C)': ['mean'], 
    #     'Humidity(%)': ['mean'], 
    #     'Wind_speed_(m/s)': ['mean'], 
    #     'Visibility_(10m)': ['mean'], 
    #     'Dew_point_temperature(°C)': ['mean'], 
    #     'Solar_Radiation_(MJ/m2)': ['mean'], 
    #     'Rainfall(mm)': ['mean'], 
    #     'Snowfall_(cm)': ['mean'], 
    #     # 'Seasons': ['mean'], 
    #     # 'Holiday': ['mean'], 
    #     # 'Functioning_Day': ['mean']
    #     })

    # dfHour = df2.groupby('Hour').agg({
    #     'Rented_Bike_Count': ['sum'], 
    #     'Temperature(°C)': ['mean'], 
    #     'Humidity(%)': ['mean'], 
    #     'Wind_speed_(m/s)': ['mean'], 
    #     'Visibility_(10m)': ['mean'], 
    #     'Dew_point_temperature(°C)': ['mean'], 
    #     'Solar_Radiation_(MJ/m2)': ['mean'], 
    #     'Rainfall(mm)': ['mean'], 
    #     'Snowfall_(cm)': ['mean'], 
    #     # 'Seasons': ['mean'], 
    #     # 'Holiday': ['mean'], 
    #     # 'Functioning_Day': ['mean']
    #     })

    # print(dfHour.head())
    # print(dfMonth.head())
    #######################################################################################################
    
    


    '''Models'''
    #######################################################################################################
    # Dependend & Independend variable
    #######################################################################################################
    # column_NotAVariable = ["Date", "Rented_Bike_Count", "Year", "Month", "Weekday"]
    # # Independent Variable
    # independent_variables = [x for x in df.columns if x not in column_NotAVariable]
    # # Dependent Variables
    # dependent_Variable = "Rented_Bike_Count"
    # # Model Build
    # Model_Build = ModelBuild(df, independent_variables, dependent_Variable)
    # # Run exhaustive search to find best indepentend variables
    # # exhaustive_search_result_Frame = Model_Build.exhaustive_search()
    
    # Model_independent_variables = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind_speed_(m/s)', 'Dew_point_temperature(°C)', 'Solar_Radiation_(MJ/m2)', 'Rainfall(mm)', 'Snowfall_(cm)', 'Seasons_2', 'Seasons_3', 'Seasons_1', 'Holiday_1', 'Functioning_Day_1']
    # Model_dependent_Variable = "Rented_Bike_Count"
  
                        #######################################################################################################
    # df2 = Bike_Data()
    # Replace " " with "_" in all columns
    # df2.columns = [s.strip().replace(" ", "_") for s in df2.columns] # all columns

    
    # df2 = df2.fillna(0)
    # column_NotAVariable = ["Date", "Rented_Bike_Count", "Year", "Month", "Weekday", "Hour", "Visibility_(10m)"]
    # Loop over the dataframe columns. If the dtype is == object, then append to list
    # columns_object_list = []
    # for col in df2.columns:
    #     series = df2[col]
    #     dtype = series.dtype
    #     if dtype == "object":
    #         columns_object_list.append(col)
    
    # For columns in columns_object_list. 
    # Get unique values. Enumerate over unique values and create new categories
    # for col in columns_object_list:
    #     unique_values = df2[col].unique()
    #     unique_category = {}
    #     for idx, val in enumerate(unique_values):
    #         unique_category[val] = idx 
    #     df2[col] = df2[col].astype('category')
    #     df2[col] = df2[col].cat.rename_categories(unique_category)
  

    # frame1 = df2[[
    #     "Temperature(°C)", 
    #     "Dew_point_temperature(°C)", 
    #     "Hour",
    #     "Humidity(%)", 
    #     "Wind_speed_(m/s)",
    #     "Solar_Radiation_(MJ/m2)",
    #     "Rainfall(mm)",
    #     "Snowfall_(cm)",
    #     "Seasons",
    #     "Holiday",
    #     "Functioning_Day"
    #     ]]

    
    # frame2 = df2.copy()
    # frame2 = frame2[[x for x in df2.columns if x not in column_NotAVariable]]
    # Rainfall and Snowfall is set to 1 if their is rain or snow
    # frame2['Rainfall(mm)'] = frame2['Rainfall(mm)'].apply(lambda x: 1 if x > 0 else 0)
    # frame2['Snowfall_(cm)'] = frame2['Snowfall_(cm)'].apply(lambda x: 1 if x > 0 else 0)
    # Humidity based on information founded on dmi
    # frame2.loc[frame2['Humidity(%)'] <= 20, 'Humidity(%)'] = 0
    # frame2.loc[frame2['Humidity(%)'] >= 50, 'Humidity(%)'] = 2
    # frame2.loc[frame2['Humidity(%)'] < 2, 'Humidity(%)'] = 1
    # Convert Wind Speed to the Beaufort scale - https://duda.dk/tema/vindstyrker/
    # frame2['Wind_speed_(m/s)'] = frame2['Wind_speed_(m/s)'].apply(lambda x: 1 if x >= 3.3 else 0)
    
    # frame3 = df2[[
    #     "Temperature(°C)", 
    #     "Hour", 
    #     "Seasons",
    #     "Holiday",
    #     "Functioning_Day"
    #     ]]

    # Setup featurs - Different column combinations - A mix of independent variables
    # X1 = pd.get_dummies(frame1, prefix_sep="_", drop_first=True)
    # X2 = pd.get_dummies(frame2, prefix_sep="_", drop_first=True)
    # X3 = pd.get_dummies(frame3, prefix_sep="_", drop_first=True)
    # features_to_include = [X1, X2, X3]
    # Y_Series = df2["Rented_Bike_Count"]


    # ######################################################################################################
    # KNN Model - https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
    # ######################################################################################################
    # StockResultDf(BestModelWithoutTraining("KNN", KNN, features_to_include, Y_Series))
    # ######################################################################################################

    # ######################################################################################################
    # Linear Regression Model
    # ######################################################################################################
    # StockResultDf(BestModelWithoutTraining("Linear Regression", LinearRegressionModel, features_to_include, Y_Series))
    # ######################################################################################################

    # ######################################################################################################
    # Use Cross Validation for finding the best hyperparameters for the first dataset
    # Create train and test set
    # ######################################################################################################
    # x_train, x_test, y_train, y_test = train_test_split(X1, Y_Series, test_size=0.6) #org 0.33

    # Scale
    # scaler = StandardScaler()
    # scaler.fit(x_train)             
    # x_train = scaler.transform(x_train, copy = False)
    # x_test  = scaler.transform(x_test, copy = False)

    # Search the hyperparameters
    # parameters = {'gamma':[0.01, 0.025, 0.05, 0.75, 0.1, 0.25, 0.5]}
    # grid = GridSearchCV(svm.SVR(), parameters, n_jobs=-1, cv=5)
    # grid.fit(x_train,y_train)
    # print('Best r2 accuracy : ' + str(round(grid.best_score_ * 100,2)) + '% with Gamma : ' + str(grid.best_estimator_))

    # Redo for finding the better gamma and a good C
    # parameters = {'gamma':[0.03, 0.04, 0.05, 0.06 ,0.07], 'C' : [0.5, 1, 2, 3, 5, 10]}
    # grid = GridSearchCV(svm.SVR(), parameters, n_jobs=-1, cv=5)
    # grid.fit(x_train,y_train)
    # print('Best r2 accuracy : ' + str(round(grid.best_score_ * 100,2)) + '% with Gamma and C : ' + str(grid.best_estimator_))

    # Determine the degree and affine C
    # parameters = {'gamma' : [0.065,0.07,0.075], 'C' : [10, 15, 20], 'degree' : [0.15, 0.25, 0.5]}
    # grid = GridSearchCV(svm.SVR(), parameters, n_jobs=-1, cv=5)
    # grid.fit(x_train,y_train)
    # print('Best r2 accuracy : ' + str(round(grid.best_score_ * 100,3)) + '% with Gamma , degree and C : ' + str(grid.best_estimator_))

    # Finish by searching the kernel
    # parameters = {'C' : [20, 50, 100], 'kernel' : ['rbf','poly','sigmoid','linear'], 'gamma' : [0.073, 0.075, 0.077], 'degree' : [0.10, 0.15, 0.20]}
    # grid = GridSearchCV(svm.SVR(), parameters, n_jobs=-1, cv=5)
    # grid.fit(x_train,y_train)
    # print('Best r2 accuracy : ' + str(round(grid.best_score_ * 100,3)) + '% with Gamma, C, degree : ' + str(grid.best_estimator_))

    # Redo this for all datasets reusing the best hyperparameters
    # params = {
    #             'C' : [20, 50, 100], 
    #             'kernel' : ['rbf','poly','sigmoid','linear'], 
    #             'gamma' : [0.073, 0.075, 0.077], 
    #             'degree' : [0.10, 0.15, 0.20]
    # }
    # StockResultDf(BestModelWithTraining('Cross Validation',GetScoreHyperparametres,svm.SVR(),params, features_to_include, Y_Series))
    # ######################################################################################################
    
    # ######################################################################################################
    # Lasso
    # ######################################################################################################
    # params = {  
    #             "max_iter"  : [ 250, 500, 1000, 1500 ],
    #             "alpha"     : [ 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0], 
    #             "selection" : ["random", "cyclic"]
    # }
    # StockResultDf(BestModelWithTraining('Lasso', GetScoreHyperparametres, Lasso(), params, features_to_include, Y_Series))
    # ######################################################################################################

    # ######################################################################################################
    # AdaBoostRegressor
    # ######################################################################################################
    # params = {
    #     "n_estimators"         : [50, 100, 150],
    #     'loss'                 : ["linear", "square", "exponential"]
    # }
    # StockResultDf(BestModelWithTraining('AdaBoost', GetScoreHyperparametres, AdaBoostRegressor(), params, features_to_include, Y_Series))
    # ######################################################################################################

    # ######################################################################################################
    # Extra trees
    # ######################################################################################################
    # params = {
    #     "n_estimators"         : [50, 100, 150],
    #     'max_depth'            : [5],
    #     'bootstrap'            : [True,False]
    # }
    # StockResultDf(BestModelWithTraining('ExtraTrees', GetScoreHyperparametres, ExtraTreesRegressor(), params, features_to_include, Y_Series))
    # ######################################################################################################

    # ######################################################################################################
    # Random Forest
    # ######################################################################################################
    # params = {
    #     "n_estimators"         : [50, 100, 150],
    #     'max_depth'            : [5],
    #     'bootstrap'            : [True,False]
    # }
    # StockResultDf(BestModelWithTraining('Random Forest', GetScoreHyperparametres, RandomForestRegressor(), params, features_to_include, Y_Series))
    # ######################################################################################################

    # ######################################################################################################
    # XGBoost
    # ######################################################################################################
    # params = {
    #     "max_depth" : [4],
    #     "gamma" : [0.077, 0.5, 0.75, 1]
    # }
    # StockResultDf(BestModelWithTraining('XGBoost', GetScoreHyperparametres, XGBRegressor(), params, features_to_include, Y_Series))
    # ######################################################################################################

    # ######################################################################################################
    # Bagging Regressor
    # ######################################################################################################
    # from sklearn.ensemble import BaggingRegressor
    # params = {
    #     "n_estimators"         : [10, 25],
    #     'bootstrap'            : [True,False]
    # }
    # StockResultDf(BestModelWithTraining('Bagging Regressor', GetScoreHyperparametres, BaggingRegressor(), params, features_to_include, Y_Series))
    # ######################################################################################################


    # phonomynol regression
    # Probit Regression
    # Gradient boosting algorithms
    # Elastic Net 

    # ResultDf.to_excel("Model_Result.xlsx")

    # print(ResultDf)
    



    
    
   




























main()
