
#packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
from datetime import date, time, datetime,timedelta
import plotly.express as px
import plotly.graph_objects as go 
import random
import pickle
import dill
import time as t
from collections import defaultdict
# %matplotlib inline



# ML Algorithms For Recommendation System 1
from sklearn.model_selection import train_test_split
from surprise import Dataset,Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import RandomizedSearchCV
from surprise import accuracy
from surprise import SVD,SVDpp,NMF,KNNBasic,KNNBaseline,KNNWithMeans,KNNWithZScore,CoClustering,BaselineOnly,NormalPredictor,SlopeOne
from surprise.model_selection import GridSearchCV
from surprise import accuracy
from surprise.model_selection import KFold
#from surprise.model_selection import train_test_split
from surprise.model_selection import PredefinedKFold

# ML Algorithms For Recommendation System 2
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k,recall_at_k
from lightfm.evaluation import auc_score,calculate_auc_from_rank
import itertools
 
#load extra data 
# products_df = pd.read_csv("./data/Products.csv")
# customers_df = pd.read_csv("./data/Customers.csv")
  

#functions

# ------------------------------------------------DATA PREPROCESSING --------------------------------------------------------------------
def id_correspondance(df,column_id):
    """ 
        Input :
        -------
            dataframe : mandatory columns are 'ProductId', 'ProductIdName' or similar 
            column_id : ProductId/ProductIdName or CustomerId/CustomerIdName
            
        Purpose :
        -------
            Corresponds ProductId to ONE ProductIdName, 
            if one ProductIdName has more than one ProductId then the most recent one is used 

            Looking into  CustomerId and CustomerIdName correspondence 
            (!! DOESN'T BOTHER ME if one CustomerIdName has more than one CustomerId , 
            so i don't change the correspondence !!)
            The change in the data DOES NOT APPLY for CustomerId     
        
        Output:
        -------
            Dataframe in which every ProductIdName has one ProductId
    """
    #match columns to ids 
    if (column_id=='ProductId') or (column_id=='ProductIdName'):
        id='ProductId'
        name='ProductIdName'
    elif (column_id=='CustomerId') or (column_id=='CustomerIdName'):
        id='CustomerId' 
        name='CustomerIdName'  
    

    print("Number of unique {0} :  {1} ".format(id,df[id].nunique()))
    print("Number of  unique {0} :  {1} ".format(name,df[name].nunique()))

    #find the ProductIdName/CustomerId with more than one ids
    no_doubles=df[[id,name]].drop_duplicates()
    #how many ids correspond to the same name 
    values=no_doubles[name].value_counts().to_frame()
    values.reset_index(inplace=True)
    values=values.rename({name:'correspondence','index':name},axis=1)
    
    #get those names with more than one id 
    more_than_one_id=values[name].loc[values['correspondence'] > 1].values
    print("{0} with more than one {1} : \n {2}".format(name,id,values.loc[values['correspondence'] > 1]))
    print('\n')

    #the change applies only to ProductId/ProductIdName
    if (column_id=='ProductId') or (column_id=='ProductIdName'):
       for product_name in more_than_one_id:

            #find the ProductId with the latest DateDelivered
            productid=df[df.ProductIdName== product_name].sort_values('DateDelivered',ascending=False)['ProductId'].values[0]
            #change the ProductId 
            df.loc[(df['ProductIdName']==product_name),'ProductId']=productid
  
       print(df[['ProductId','ProductIdName']].nunique())
 
    pass

def filter_similarity(df1,df2,column):
        """
        Input:
        ------
                df1:        one of the dataframes to do the filtering 
                df2:        second dataframe 
                column:     feature by which i am going to filter
         Purpose:
        --------
                Keep the same rows according to the column the user provides (CustomerId/ProductId) in both dataframes
        Output:
        -------
                Updated dataframes
                Information: how many rows where affected in each dataframe
        """
        print('Keep the same rows filtering:\n')
        print('Number of rows in first dataframe before: {}'.format(df1.shape[0]))
        print('Number of rows in second dataframe before: {}'.format(df2.shape[0]))
        print('Number of unique {0} rows in first dataframe: {1}'.format(column , df1[column].nunique())) 
        print('Number of unique {0} rows in second dataframe: {1}'.format(column , df2[column].nunique()))
 
        #find rows that exist in both dataframes
        same=set(df1[column]) & set(df2[column])
        #keep only the rows that exist in both dataframes 
        df1=df1.loc[df1[column].isin(same)]  
        df2=df2.loc[df2[column].isin(same)] 
        print('Number of {0} rows that exist in both dataframes: {1}\n'.format(column , len(same)))
        return df1,df2
        
def filter_invoices(df,threshold):
    """
    Input:
    ------
            df1:        first dataframe 
            threshold:   MINIMUM number of invoices per customer)

    Purpose:
    --------
            Keep those customers from  dataframe whose unique invoices ARE EQUAL OR 
            GREATER than the rovided threshold 
    Output:
    -------
            Updated dataframes
            Information about dataframes' rows 
    """
    print('Invoice per customer filtering:\n')
    print('Number of rows before: {}'.format(df.shape[0]))
    #keep only customers who satisfy the threshold
    above_threshold=df.groupby('CustomerId')['invoiceid'].nunique() >= threshold
    #customers who satisfy the threshold
    above_threshold=above_threshold[above_threshold]  
    #filter datafame 
    df=df.loc[df['CustomerId'].isin(above_threshold.index)]
    print('Number of customers with more than {0} invoices: {1}'.format((threshold-1),above_threshold.size))
    print('Number of rows after: {}\n'.format(df.shape[0]))
    return df

def fix_quantity(df):
    """
    Input:
    -----
            df:dataframe
    Purpose:
    -------
            modify the Quantity column as to represent the actual number of products per invoice
            ex. invoiceid|ProductId|Quantity
                1        |A        |1.0
                1        |A        |1.0
            to 
                1        |A        |2.0
    Output:
    ------
            Updated dataframe                     
    """
    print('Rows before: {}'.format(df.shape[0]))
    #calculate quantity per product in invoice
    new_quantity=df.groupby(['invoiceid','ProductId'])['Quantity'].sum().reset_index()
    #update data 
    df=df.drop_duplicates(['invoiceid','ProductId']).merge(new_quantity,on=['invoiceid','ProductId'])
    df=df.drop(columns='Quantity_x').rename(columns={'Quantity_y': 'Quantity'})
    print('Rows after: {}'.format(df.shape[0]))
    return df

# def long_tail(data,graph=False,title='',color='red'):
#     """
#     Input:
#     ------
#             data:  dataframe to compute most popular products (purchasing frequency)
#             graph: if True returns interactive long tail plot
#             color: specify barplot's color
#             title: plot's title 
#     Purpose:
#     --------
#             computes how many customers bought a  product  
#             (long tail plot)
#     Output:
#     -------
#             dataframe with purchasing product frequency and long tail plot               
#     """

#     long_tail=data.groupby('ProductIdName')['CustomerId'].count().reset_index()
#     long_tail=long_tail.sort_values('CustomerId',ascending=False).rename(columns={'CustomerId':'Customers'})
#     if graph == True:
#         fig = px.line(data_frame=long_tail,x='ProductIdName', y='Customers',title=title,width=800, height=700)
#         fig.add_bar(x=long_tail['ProductIdName'], y=long_tail['Customers'],marker_color=color,name='Product')
#         fig.show()
#     return long_tail

def long_tail(data,graph=False,title=''):
    """
    Input:
    ------
            data:   dataframe to compute most popular products (purchasing frequency)
            graph:  if True returns interactive long tail plot
            color:  specify barplot's color
            title:  plot's title
            choose: by sales or customers 
    Purpose:
    --------
            computes how many sales/transactions belong to each product  
            (long tail plot)
    Output:
    -------
            dataframe with purchasing product frequency and long tail plot               
    """
    #sales per product
    sales=data.groupby('ProductIdName')['CustomerId'].count().reset_index()
    sales=sales.sort_values('CustomerId',ascending=False).rename(columns={'CustomerId':'Sales'})

    #customers per product
    customers=data.groupby('ProductIdName')['CustomerId'].nunique().reset_index()
    customers=customers.sort_values('CustomerId',ascending=False).rename(columns={'CustomerId': 'Customers'})

    long_tail=sales.merge(right=customers,how='inner',on='ProductIdName')  
        
    if graph == True:
        
        fig = px.line(data_frame=long_tail,x='ProductIdName', y='Sales',title=title,width=800, height=700)
        fig.add_bar(x=long_tail['ProductIdName'] , y=long_tail['Sales'] ,marker_color='red',name='Sales',text=long_tail['Sales'],textposition='inside')
        fig.add_bar(x=long_tail['ProductIdName'], y=long_tail['Customers'],marker_color='blue',name='Customers',text=long_tail['Customers'],textposition='inside')
        
        fig.update_layout(barmode='group')
        fig.show()
    return long_tail 

def calculate_frequency(data,column='invoiceid'):
    """
    Input:
    ------
            data:   customer , product transactions data 
            column: column by which to calculate frequency (by Invoice or by Qunatity)
    Purpose:
    --------
            calculate frequency based on provided column
    Output:
    -------
            dataframe with recommedation metric 
    """
    
    #Ιnvoices/Quantity per customer and product
    count_per_customer=data.groupby(['CustomerId','ProductId'])[column].count().reset_index()
    #all invoices/product pieces per customer
    all_count=data.groupby(['CustomerId'])[column].nunique().reset_index()
    #merge
    frequency=count_per_customer.merge(right=all_count,on='CustomerId',how='inner').rename(
                          columns={'{0}_x'.format(column):'per_product_count','{}_y'.format(column):'sum'})
    #compute frequency
    frequency['frequency']=round(frequency['per_product_count'] / frequency['sum'],2)
    frequency=frequency[['CustomerId','ProductId','frequency']]
    return frequency

def reduce_bias(data,rating_column):
    """
    Input:
    ------
            data:          dataframe in RS format  
            rating_column: rating metric column
    Purpose:
    --------
            creates new rating column: by calculating the AVERAGE RATING PER PRODUCT and
            substracting it from the rating column  
    Output:
    -------
            dataframe with normalized ratings          
    """
    #calculate average 
    product_mean=data.groupby('ProductId')[rating_column].mean().reset_index().rename(
        columns={rating_column:'mean_frequency'})
    #round up mean frequency
    product_mean['mean_frequency']=round(product_mean['mean_frequency'],3)
    #add to frame 
    data=product_mean.merge(right=data,how='right',on='ProductId')
    #substract the average from the rating metric
    data['new_{}'.format(rating_column)]=round(abs(data[rating_column] - data['mean_frequency']),3)
    #get new rating's column name
    rating_column=data.columns[-1]
    return data , rating_column



def id_to_name(sample,products_df,customers_df,choose='products'):
    """
    Input:  
    ------
            sample       : a list/array 
            products_df  : dataframe with all the product ids and names 
            customers_df : dataframe with all the customer ids and names
            choose       : 'product','p' or 'customer','c' depending on converting product or customer ids            

    Purpose: 
    -------
            convert inputted ids to names (products/customers) or the opposite
    Output:  
    -------
            a list of names (products/customers) or ids
    """
 

    if choose in ['products','p','P','Products']:
        if sample[0] in products_df['ProductID'].values:
            ids=[]
            for id in sample:
                #match id to name
                ids.append(products_df.loc[products_df['ProductID'] == id]['productName'].values[0])
            return ids
        else:
            names=[]
            for name in sample:
                #match name to id 
                names.append(products_df.loc[products_df['productName'] == name]['ProductID'].values[0])
            return names  
 
    elif choose in ['customers','c','C','Customers']  :
        if sample[0] in customers_df['contactID'].values:
            ids=[]
            for id in sample:
                ids.append(customers_df.loc[customers_df['contactID'] == id]['fullName'].values[0])
            return ids

        else:
            names=[]
            for name in sample:
                names.append(customers_df.loc[customers_df['fullName'] == name]['contactID'].values[0])
            return names
# ------------------------------------------------END OF DATA PREPROCESSING--------------------------------------------------------------------

# -------------------------------------------------------RS MODEL 1--------------------------------------------------------------------


def load_dill(rating_column,load_cv=False):
    """
    Input:
    ------
            rating_column:rating column for RS
            load_cv:      if True returns cv results depending to rating column input
    Purpose:
    --------
           load dill train and test dataset,
           undergone filtering or cv results
           
    Output:
    -------
           train and test dataframe in RS format of retrieve
           cv results 
    """
    if rating_column in ['Quantity','quantity','Q','q']:
        train_file='train_q.d'
        test_file='test_q.d'
        cv_file='CV_Quantity.d'
        
    elif rating_column in ['Frequency1','frequency1','F1','f1']:
        train_file='train_f1.d'
        test_file='test_f1.d'
        cv_file='CV_Frequency1.d'

    elif rating_column in ['Frequency2','frequency2','F2','f2']:
        train_file='train_f2.d'
        test_file='test_f2.d'
        cv_file='CV_Frequency2.d'
        
        
    elif rating_column in ['Bias1','bias1','B1','b1','new_Frequency1']:
        train_file='train_bias1.d'
        test_file='test_bias1.d'
        cv_file='CV_new_Frequency1.d'
        
    elif rating_column in ['Bias2','bias2','B2','b2','new_Frequency2']:
        train_file='train_bias2.d'
        test_file='test_bias2.d'
        cv_file='CV_new_Frequency2.d'        
        
    if load_cv==False:    
        with open(train_file ,'rb') as f:
            train= dill.load(f)
        with open(test_file,'rb') as f:
            test=dill.load(f)
    
        # #get new rating's column name 
        # rating_column=train.columns
    
        return train, test , rating_column
    else:
        with open(cv_file,'rb') as f:
            results_cv=dill.load(f)
        return results_cv               



def read_data(data,rating_column,trainset=False,anti_test=False):
    """
    Input:
    ------
            data:          dataframe in RS format (['CustomerId,'ProductId','Rating'])
            rating_column: name of ratings column 
            trainset:      transforms Dataset  data to Trainset Surprise mode
            anti_test:     creates set with all the products which the customers has NOT bought as 
                           to recommend ONLY NEW products
    Purpose:
    --------
            tranforms to Surprise Dataset,Trainset and anti_test mode 
    Output:
    ------- 
            dataframe in Dataset mode (for Cross Validation / GridSerach)
            Trainset object (fit())
            Anti_testset object (test())              
    """

    #data=data[['CustomerId','ProductId',rating_column]]
    mn=data[data.columns[-1]].min()
    mx=data[data.columns[-1]].max()
    #surpise Readeer
    reader=Reader(line_format='user item rating',rating_scale=(mn,mx))

    rating_column=data.columns[-1]
    #to Dataset mode  
    data=Dataset.load_from_df(data,reader)
    if trainset == False & anti_test == False:
        return data,rating_column

    #to trainset 
    if trainset==True:
        trainset=data.build_full_trainset()
        return trainset
    #create anti-testset    
    if anti_test==True:
        anti_testset=trainset.build_anti_testset()
        return  anti_testset


def calculate_sparsity(data,rating_column,products_df,customers_df,matrix_display='id',
                       compressed=True):
        """
        Input:
        ------
                data:            data to create sparse matrix
                rating_column:   column with  ratings
                                sample:          a list/array
                products_df:     dataframe with all the product ids and names
                customer_df:     dataframe with all the customer ids and names
                matrix_display:  display per id or name customers and products
                #only_sparsity:   if True it just returns sparsity of matrix
                                 #(IGNORES compressed input)|
                                 #perform two seperate function calls)
                compressed:      if True returns matrix in csr_matrix mode 
                                 otherwise as sparse matrix
                choose:          'product','p' or 'customer','c' depending on converting product or customer ids
                                
        Purpose:
        -------
                create sparse matrix and count sparsity of matrix
                Spasrsity : 1 - (non zero values / total matrix elements)
        Output:
        ------
                sparse matrix ,sparsity value        
        """
        
        #dictate axes for table 
        if matrix_display in ('i', 'id','ID')  :
            data=data[['CustomerId','ProductId',rating_column]]
            
 
            
        elif matrix_display in ('n','name','names','N')  :
           
            #get names
            ratings=data[rating_column]
            product_name=id_to_name(sample=data['ProductId'].values,
                                    products_df=products_df,customers_df=customers_df,choose='products')
            customer_name=id_to_name(sample=data['CustomerId'].values,
                                     products_df=products_df,customers_df=customers_df,choose='customers')

            
            data = pd.DataFrame()
            data['CustomerIdName']=customer_name
            data['ProductIdName']=product_name
            data[rating_column]=ratings
            
        #sparse matrix
        sparse_matrix=data.pivot_table(index=data.columns[0],columns=data.columns[1],
                                   values=rating_column).fillna(0)
        
        #calculate sparsity
        
        #known raitng values (not NaN matrix values)
        known_raitngs = data.shape[0]
        #total matrix elements
        all_matrix_elements = sparse_matrix.size
        sparsity = round((1.0 - known_raitngs / all_matrix_elements),2)
        print('Sparsity: {}'.format(sparsity))
 
            
        if compressed == True:
              
            return  csr_matrix(sparse_matrix.values)
        else:
            return sparse_matrix 

def Leave_highest_rating_out(dataframe):
    """
    Input:
    ------
            dataframe : dataframe used as training set 
    Purpose:
    -------
            removes from the  dataframe each consumer's highest rated product 
    Output:
    ------
            dataframe with the removed rows                
    """
    rating_column=dataframe.columns[-1]
    print('Rows before: {} \n'.format(dataframe.shape[0]))
    
    #sort customer,ratings
    dataframe=dataframe.sort_values(['CustomerId',rating_column],ascending=False)

   #get highest rating row per customer
    highest_rated_index=dataframe.groupby('CustomerId').apply(
                        lambda first_rating: first_rating.index[0]).reset_index()[0].values
    #locate them
    highest_rated_products=dataframe.loc[highest_rated_index]
    # drop them 
    dataframe.drop(index=highest_rated_index,inplace=True)
    print('Rows after: {} \n'.format(dataframe.shape[0]))

    return dataframe,highest_rated_products,rating_column 
    
#--------------------------------------------- Cross Validation / Hyperparameter tunning -----------------------------------------------    

def cross_validation(data,algorithm,measures=['rmse'],cv=3,return_train_measures=True,verbose=False,n_jobs=1):
    """
    Input:
    ------
            data:                  data for cross validation in surpise Dataset mode
            algoritm:              list of algorithms (example: [SVD(),..,])
            measures:              error metrics ['rmse','mae']
            cv:                    cross validation splits 
            return_train_measures: to count or not error metrics for training set
            verbose:               print or not extra info
            n_jobs:                CPUs to use            

    Purpose:
    --------
            execute cross validation in provided data 
    Output:
    -------
            returns cv results as dataframe indexed by algorithm name and ordered 
            by error metric score for test set               
    """
    tmp=[]
    results=[]
    start= t.time()
    for algo in algorithm: 
        
        #cross validate for every algorithm that is in the provided list
        cv_results=cross_validate(data=data,algo=algo,measures=measures,cv=cv,
                                  return_train_measures=return_train_measures,verbose=verbose,n_jobs=n_jobs)

        tmp=round(pd.DataFrame.from_dict(cv_results).mean(axis=0),3)
        tmp=tmp.append(pd.Series(str(algo).split(' ')[0].split('.')[-1],index=['Algorithm']))
        results.append(tmp)
    finish=t.time() - start

    #handle NMF exception
    if len(algorithm)==10:
        results.append(pd.Series([0,0,0,0,'NMF'],index=['test_rmse','train_rmse','fit_time','test_time','Algorithm']))
     
    return pd.DataFrame(results).set_index('Algorithm').sort_values('test_rmse') , finish

#                           ----------------------------------HYPERPARAMETER TUNNING-----------------------------------------

    
def get_paramgrid(algo):
    """
    Input:
    ------
            algo:provide specific algorithm name 1
    --------
            according to the iputted algo returns different 
            hyperparameters for algorithm tunning
    Output:
    -------
            dictionary with range of parameters

    """
    if algo =='SVD' or algo == 'SVDpp':

        param_grid1= {'n_factors':[10,20,30],
                      'n_epochs':[5,10,20],
                      'lr_all':np.logspace(-4,-2,10),  
                      'reg_all':[0.4,0.6]}

        param_grid2 = {'n_factors':[10,20],
                       'n_epochs':[5,10],
                       'lr_all':np.logspace(-5,-4,5),
                       'reg_all':[0.4,0.6]}
        params=[]
        params = param_grid1,param_grid2
        #randomize the param_grid choice
        return random.choice(params)

    elif algo == 'NMF':
        return {'n_factors':[10,20,30],
                'n_epochs':[5,10,20]}

    elif algo == 'BaselineOnly':
        param_grid1= {'bsl_options':{'method': ['sgd'], 
                                     'learning_rate': np.logspace(-3,-5,7), #np.logspace(-3,-4,5)
                                     'n_epochs':[15,20,30]
                      }}
        param_grid2 = {'bsl_options':{'method':['als'],
                                      'random_state':[250],
                                      'n_epochs':[5,10],
                                      'reg_u':[10,12,15],
                                      'reg_i':[5,10,15] 
                      }}
        params=[]
        params=param_grid1,param_grid2
        #randomize the param_grid choice
        return random.choice(params)
    
    elif algo == 'KNNBasic' or algo == 'KNNWithMeans' or algo == 'KNNWithZScore':
        return {'k': [10, 15,20],

                'sim_options': {'name': ['msd', 'cosine','pearson'],
                                'min_support': [1, 5],
                                'user_based': [True]}} 
    elif algo == 'KNNBaseline':
        return {'bsl_options': {'method': ['als', 'sgd'],
                                'learning_rate': np.logspace(-5,-6,5),
                                'reg': [1, 2]},
                                
                                'k': [3, 10],

                'sim_options': {'name': ['msd', 'cosine'],
                                'min_support': [1, 5],
                                'user_based': [True]}
                 }
    elif algo == 'CoClustering':
        return  {'n_cltr_u':[3,5],
                 'n_cltr_i':[3,4,5],
                 'n_epochs':[10,20,30],
                 
                 
                 'random_state':[0]}  
                 


def Grid_Search(data,algorithm,param_grid,Grid_Search=True,cv=3,measures=['rmse'],joblib_verbose=10,n_iter=10,n_jobs=1):
    """
    Input:
    ------
            data:           data in surprise Dataset mode
            
            algorithm:      the name of the algorithm (object)
            
            
            Grid_Search:    if True does GridSearchCV if False RandomizedSearchCV
            
            param_grid:     provide dict with parameters for tunning 
            measures:       string or least of erros measures ['rmse','mae']
            cv:             number of dataset splits default is 3 (here)
            joblib_verbose: number of printed messages during the Grid/Randomized search 
            n_iter:         number of iterations (only for RandomizedSearchCV)
                            (The number of parameter settings that are tried is given by n_iter)
            n_jobs:         CPUs to use                
    Purpose:
    --------
            Executes GridSearchCV or RandomizedSearchCV for parameter tunning        
    Output:
    -------
            Returns best score , best parameters , dataframe with information
            for each search run
                  
    """
    start=t.time()
    if Grid_Search==True:
        gs=GridSearchCV(algorithm,param_grid,measures,cv,n_jobs,joblib_verbose)
        gs.fit(data)
        score=gs.best_score
        param=gs.best_params
        results=pd.DataFrame(gs.cv_results)
        results['Search']='GridSearchCV'
    else:
        rs=RandomizedSearchCV(algorithm,param_grid,n_iter,measures,cv,n_jobs,joblib_verbose)
        rs.fit(data)
        score=rs.best_score
        param=rs.best_params
        results=pd.DataFrame(rs.cv_results)
        results['Search']='RandomizedSearchCV'
        #algorithm=rs.best_estimator_      get tunned with best parameters algo         
    timimg=t.time() - start 
    results['algo']=algorithm
    #tune algo to the best paramaters and return it 
    algorithm=algorithm(**param['rmse'])
    #results=results.sort_values('mean_test_rmse')

    return score , param, timimg, results,algorithm         
     
#                           ----------------------------------TRAINING -----------------------------------------------------
 
def get_predictions(data,algo,anti_test=True,provided_test=None):
    """
    Input:
    -----
            data:           dataframe with these columns in order [['CustomerId,'ProductId','Ranking']] or Dataset object
            algo:           algorithm to do the fitting (provide specific hyperparameters otherwise default
                            will be used)
            anti_test :     predictions for prodcucts that the customer hasn't purchased 
            provided_test:  predictions for all the prodcucts in the test set 
                            (when i want to check algorithm's prediction efficiency)
    Purpose:
    --------
            Fits the algorithm to the data and computes the predictions for 
            (only products the user has not bought)
            
    Output:
    -------
            returns predictions (list)
    """
    #if isinstance(data,pd.DataFrame) :
         #transform to surpise format 
        #(data,rating_column)=read_data(data,rating_column=provided_test.columns[-1])
    
    start=t.time()    
    trainset=data.build_full_trainset()
    #fit algo 
    algo.fit(trainset)
    if anti_test == True:
        #prediction for those products that do not exist on trainset
        
        #create testset based on not rated trainset's products
        testset=trainset.build_anti_testset()
        predictions=algo.test(testset)
        fit_time=t.time() - start
        return predictions,fit_time

    else:
        #prediction on provided testset
        
        #test data to surpise mode
        (data_test,rating_column)=read_data(provided_test,rating_column=provided_test.columns[-1])
        #build testset 
        to_trainset=data_test.build_full_trainset()
        testset=to_trainset.build_testset()
        #get predicitons
        predictions=algo.test(testset)
        #check accuracy 
        test_score=accuracy.rmse(predictions)
        fit_time=t.time() - start

        return predictions,fit_time,test_score

def training_log(rating_column,algo,fit_time='-',create_log=False,info=pd.DataFrame(),leave_one_out=False):
        """
        Input:
        ------  rating_column: specifies the rating type 
                algo:          algorithm used at time 
                fit_time:      training elapsed time 
                create_log:    if True ovewrites the existent dataframe(log) and creates a new one 
                               if False adds new row to the existent one 
                info:          dataframe with (hyperparameter) tunning information
                Leave_one_out: if True, training data with highest rated product per customer removed 
                               (for evaluation purposes)

        Purpose:
        --------
                creates dataframe with log-like purpose (training history)  
        Output:
        -------
                dataframe with aggregatted information regarding training trails
        """         

        if create_log == True :
                #create dataframe and store it to disk
                log=pd.DataFrame(columns=['Date','Data','Algo','Tunning','Tunning_time','Best_parameters',
                                          'Tunning_RMSE','Train_time','Leave_one_out'])
                dill.dump(log,open('train_log.d','wb'))  
        else:
                with open('train_log.d','rb') as f:
                        log=dill.load(f)
 
        date=datetime.now()
        data_type=rating_column
        if type(algo) != str :
                algo=str(algo).split()[0].split('.')[-1]

        if info.empty==True:
                tunning='-'
                tunning_time='-'
                best_parameters='-'
                tunning_score='-'
 
        else:
                info=info.sort_values('mean_test_rmse').head(1)
                tunning=info['Search'].values[0]
                tunning_time=info['mean_fit_time'].values[0]
                best_parameters=info['params'].values[0]
                tunning_score=info['mean_test_rmse'].values[0]
         
 
        train_info=[date,data_type,algo,tunning,tunning_time,best_parameters,
                    tunning_score,round(fit_time,2),leave_one_out]
        #add values to df
        log=log.T
        log[log.shape[1] + 1] = train_info
        log=log.T
        #update dill file 
        dill.dump(log,open('train_log.d','wb'))
        return log   



 
  

#                        ----------------------------------GET RECOMMENDATIONS--------------------------------------------------
 

def recommend(products_df,customers_df,predictions,N=5):
    """
    Input:
    ------
            products_df:  dataframe with all the product ids and names (for id_to_name())
            customers_df: dataframe with all the customer ids and names (for id_to_name())
            predictions:  algorithm prediction output 
            N:            number of recommendation per customer
    Purpose:
    --------
            computing the topN product recommendation for the customers
    Output:
    -------
            dataframe with topN product recommendation(as a list) per customer       
    """
    preds=pd.DataFrame(predictions,
                      columns=['CustomerId','ProductId','Real_value','Prediction','Details'])
    preds=preds.sort_values(['CustomerId','Prediction'],ascending=False).groupby('CustomerId')

    top_id=[]
    top_name=[]
    for customer in preds.groups.keys():
        high_products=preds.get_group(customer)['ProductId'].head(N).values
        top_id.append(high_products)
        top_name.append(id_to_name(high_products,products_df,customers_df,'products'))
    recommendation=pd.DataFrame(index=preds.groups.keys(),
                                columns=['CustomerName','Top{}'.format(N)])
    recommendation['CustomerName']=id_to_name(list(preds.groups.keys()),products_df, customers_df, choose='customers')
    recommendation['Top{}_Ids'.format(N)]=top_id
    recommendation['Top{}'.format(N)]=top_name
    return recommendation,N

def find_matches(products_df, customers_df,list1,list2,return_names=True):
    """
    Input:
    ------
            products_df:  dataframe with all the product ids and names (for id_to_name())
            customers_df: dataframe with all the customer ids and names (for id_to_name())
            list1:        smaller list of the two 
            list2:        second list to compare
            threshold:    minimum number of matches to consider hit
            return_names: If True , returns the Names corresponding to 
                          every Id (Products or Customers)
    Purpose:
    --------
            compare two lists and returns the matched elements 
    Output:
    -------
            number of matched items , list with the names of the matched items 
    """
 
    #complute intersection between the sets
    intersection = set(list1) & set(list2)
    if intersection :

        if return_names == True:
 
            #store intesection prodcuct names
            matched_products=(id_to_name(list(intersection), products_df, customers_df, choose='products'))

        else:

            matched_products=intersection
 
        #measure how many products matched 
        matches=len(intersection)
    else:
         matches=0
         matched_products=[]   

    return matches, matched_products


def recommendation_results(customer,recommendation,train,test,products_df,customers_df,N=5,return_values=False,show=False):
    """
    Input:
    ------
                customer:       customerId
                recommendation: dataframe with topN recommendations per customer 
                train:          training dataframe (prior purchases)
                test:           dataframe with future purchases
                products_df:    dataframe with all the product ids and names (for id_to_name())
                customers_df:   dataframe with all the customer ids and names (for id_to_name())
                N:              number of recommendations in TopN
                return values:  if True composes dictionary with customer recommendation results
                                (TopN recommendations,Hits,Future purchases,Past purchases,New products bought)
                show:           prints recommendation results (TopN recommendations,Hits,Future purchases,
                                Past purchases,New products bought)


    Purpose:
    -------
            computing recommendation results PER CUSTOMER
            (TopN recommendations,Hits,Future purchases,Past purchases,New products bought)
    Output:
    -------
            returns dictionary with summary recommendation results per customer 
            can also print recommendation results per customer
    """
    #get customer's topN, prior purchased products, future purchased
    to_recommend=recommendation.loc[customer]['Top{}_Ids'.format(N)]
    will_buy=test.loc[test['CustomerId']==customer]['ProductId'].values
    #new products not in trainset only in test
    #new_products
    bought=train.loc[train['CustomerId']==customer]['ProductId'].values
    
    #check for hits in TopN , future purchases (hit rate)
    recommendation_matches=find_matches(products_df=products_df, customers_df=customers_df,list1=to_recommend,list2=will_buy,return_names=True)
    #TopN , prior purchases
    are_bought_matches=find_matches(products_df=products_df, customers_df=customers_df,list1=to_recommend,list2=bought,return_names=True)
    
    #products in future purchases but not in prior ones (find set difference)
    new_products=set(will_buy) - set(bought)
    if len(new_products) == 0:
        new_products_names=[]
    else:
        new_products_names=id_to_name(sample=list(new_products), products_df=products_df, customers_df=customers_df, choose='products')
        

    if show == True:
        print('CustomerId: {} , Customer Name: {}'.format(customer,id_to_name(sample=[customer], products_df=products_df, customers_df=customers_df, choose='customers')[0]))
        # print('\n')
        print('Recommendations:\n {}'.format(id_to_name(sample=to_recommend, products_df=products_df, customers_df=customers_df, choose='products')))
        # print('\n')
        print('Will buy:\n {}'.format(id_to_name(sample=will_buy, products_df=products_df, customers_df=customers_df, choose='products')))
        # print('\n')
        print('Hits:\n {} {} '.format(recommendation_matches[0],recommendation_matches[1]))
        # print('\n')
        print('Prior Purchases:\n {}'.format(id_to_name(sample=bought, products_df=products_df, customers_df=customers_df, choose='products')))
        # print('\n')
        print('Prior Purchases in Recommendations:\n {}'.format(are_bought_matches))
        # print('\n')
        print('New Products: {} \n {} '.format(len(new_products),new_products_names))
        print('\n')
 
    #create dict
    if return_values==True:
        customer_info={}
        from collections import defaultdict
        #customer_info=defaultdict(lambda: 0)
        customer_info=defaultdict(int)
        customer_info['CustomerId']=customer
        customer_info['CustomerName']=id_to_name(sample=[customer], products_df=products_df, customers_df=customers_df, choose='customers')[0]
        customer_info['Recommendations']=id_to_name(sample=to_recommend, products_df=products_df, customers_df=customers_df, choose='products')
        customer_info['Will_buy']=id_to_name(sample=will_buy, products_df=products_df, customers_df=customers_df, choose='products')
        customer_info['Hits_number']=recommendation_matches[0]
        customer_info['Hits']=recommendation_matches[1]
        customer_info['Prior_purchases']=id_to_name(sample=bought, products_df=products_df, customers_df=customers_df, choose='products')
        customer_info['Prior_and_recommended_number']=are_bought_matches[0]
        customer_info['Prior_and_recommended']=are_bought_matches[1]
        customer_info['New_purchases_number']=len(new_products)
        if len(new_products)!= 0:
            customer_info['New_purchases']=new_products_names
        return customer_info
            
    else:
       pass               
#-------------------------------------- EVALUATION ---------------------------------------------------

def calculate_accuracy(recommend,train,test,products_df,customers_df,threshold=1,N=5,show=False):
    """
    Input:
    ------
            recommend:    dataframe with TopN recommendation per customer 
                          (recommend() output)
            train:        training dataframe (prior purchases) (for recommendation_results())
            test:         dataframe with future purchases  (for recommendation_results())
            products_df:  dataframe with all the product ids and names (for id_to_name())
            customers_df: dataframe with all the customer ids and names (for id_to_name())
            threshold:    number of recommended products that got bought as to count as a 
                          succesful recommendation
            N:            number of recommendations in topN              
            show:         prints recommendation results 
                          (TopN recommendations,Hits,Future purchases,Past purchases,New products bought)
            
    Purpose:
    --------
            computes:
                    hit rate: hits / overall customers 
                             (hit:               if at least one of the TopN recommended products will be bought)
                             (overall customers: all customers: customers who bought at least one new product
                                                 from their prior purchases)
                    
                    all hits: number of hits from all the customers
                    finish :  time elapsed
    Output:
    -------
            returns accuracy aka hit rate, all hits , overall customers , time elapsed,
            customer recommendation information in dictionray format              
    """

    all_hits=0
    all_customers=0
    hit_products=[]
    customer_counter=0
    start=t.time()
    list_of_dictionaries=[]

    for customer in recommend.index:
        customer_counter += 1
        # print(customer_counter)

        #create dicitonary with recommendation results per customer
        customer_results=recommendation_results(customer=customer,recommendation=recommend,train=train,test=test,products_df=products_df,
                                                customers_df=customers_df,N=N,return_values=True,show=False)
        #store to list                                        
        list_of_dictionaries.append(customer_results)
        print(N)

        if customer_results['Hits_number'] >= threshold:
            all_hits += 1
            #store hit products
            [hit_products.append(product) for product in customer_results['Hits']]
        if customer_results['New_purchases_number'] > 0:
            all_customers += 1                
    accuracy=round(all_hits / all_customers , 2)
    finish=t.time() - start
    print('Time:'.format(finish))
    print('Accuracy:'.format(accuracy))

    return accuracy , all_hits , all_customers, finish , hit_products ,threshold , list_of_dictionaries

def precision_recall_at_N(recommendation_results,count_customers,products_df,customers_df,N=5 ):
    """
        Input:
        ------
            recommendation_results : dictionary with recommendation results per customer
            count_customers:         customers (number) with new purchases
            products_df:             dataframe with all the product ids and names (for id_to_name())
            customers_df:            dataframe with all the customer ids and names (for id_to_name())
            N:                       number of recommendation per customer

        Purpose:  Calculates Precision@N and Recall@N (and then their average)
        --------

        Output:
        -------
                Average precision@N and Recall@N
    """
    k=N
    calculate_precision=0
    precision=0
    calculate_recall=0
    recall=0

    precision_dict=defaultdict(int)
    recall_dict=defaultdict(int)


    for customer_by_number in range(0,len(recommendation_results)):

        #recall equals  hits / new purchases
        #relevant products (will buy products)
        will_buy_new_products=recommendation_results[customer_by_number]['New_purchases']
        #recommendation list
        topN=recommendation_results[customer_by_number]['Recommendations']
        #in case of no new purchases
        if will_buy_new_products==0:
            calculate_recall=0
            # recall=recall + calculate_recall
            
            continue

        #find matches between will_buy and topN aka Hits 
        (matches, matched_products) = find_matches(products_df=products_df,customers_df=customers_df,list1=topN,list2=will_buy_new_products,return_names=False)

        #recall per customer
        calculate_recall = matches / len(will_buy_new_products)
        #add to recall dict , key : CustomerId
        recall_dict[recommendation_results[customer_by_number]['CustomerId']]=calculate_recall
        #sum recall  
        recall= recall + calculate_recall  

        #precision per customer
        calculate_precision= matches / k
        #dict key: CustomerId
        precision_dict[recommendation_results[customer_by_number]['CustomerId']]=calculate_precision
        #sum precision 
        precision=precision + calculate_precision
        
        
    #get average   
    precision_avg=round(precision /  count_customers ,2)   
    recall_avg=round(recall / count_customers ,2)
    print(precision_avg, recall_avg)

    return precision_avg,  recall_avg, precision_dict, recall_dict



def product_popularity_in_recommendations(recommendation_df):
    """
    Input:
    ------
            recommendation_df: dataframe with topN for each customer
    Purpose:
    --------
            conducts after recommendation evaluation: 
            1.counts all the unique prodicts in all TopN recommendations
            2.calculates product frequency 
    Output:
    -------
            list with ['ProductName','frequency']               
    """
    all_the_recommendations=[]
    recommendations=recommendation_df['Top5']
    for recommendation_list in recommendations:
        for product in recommendation_list:
            #create a list with all the recommendations for every customer
            all_the_recommendations.append(product)
            
    #product frequnecy in recommendations as a list 
    recommendation_frequency=np.array(np.unique(all_the_recommendations,return_counts=True)).T
    #count the different products in all the topN recommendation 
    unique_products=len(recommendation_frequency)
    print('Unique products in all TopN: {} '.format(unique_products))  
    return recommendation_frequency 




def evaluation_log(rating_column,algo,N,precision_avg,recall_avg,hit_rate,hits,hit_threshold,
                    count_customers,time_elapsed,unique_TopN,unique_products,loo_rate='-',leave_one_out=False,create_log=False):
    """
    Input:
    -----
            rating_column:   specifies the rating type
            algo:            algorithm used at time 
            N:               number of recommendation for each user
            recall:          recall@N average 
            precision:       presicion@N  average   
            hit_rate:        hits / overall customers 
                             (hit:               if at least one of the TopN recommended products will be bought)
                             (overall customers: customers who bought at least one new product
                                                 from their prior purchases)
            hits:            number of hits from all the customers
            count_customers: overall customers (customers who bought at least one new product
                                                from their prior purchases)
            time_elapsed:    time passed 
            create_log:      if True ovewrites the existent dataframe(log) and creates a new one, 
                             if False adds new row to the existent one 

      
    Purpose:
    -------
            creates/updates existent log-like dataframe with evaluation progress
            preview: fsdff|fsdffs|dsfsdfsfd|SDFSDFSD|
    Output:
    -------
            dataframe with eavluation progress           
    """
    if create_log==True:
            log=pd.DataFrame(columns=['Date','Data','Algo','Leave_one_out','N','Precision@N','Recall@N','Hit_rate','Hits','Hit_threshold',
                                      'Count_customers','Compute_time','Unique_TopN','Unique_Products','LOO_Rate'])
            dill.dump(log,open('evaluation_log.d','wb'))  
    else:
            with open('evaluation_log.d','rb') as f:
                        log=dill.load(f)
 
    date=datetime.now()
    data_type=rating_column
    if type(algo) != str :
            algo=str(algo).split()[0].split('.')[-1]

 
  
    evaluation_info=[date,data_type,algo,leave_one_out,N,precision_avg,recall_avg,hit_rate,hits,hit_threshold,
                     count_customers, round(time_elapsed,2), round(unique_TopN/100,3),round(unique_products/100,2), loo_rate]
    #add values to df
    log=log.T
    log[log.shape[1] + 1] = evaluation_info
    log=log.T
    #update dill file 
    dill.dump(log,open('evaluation_log.d','wb'))
    return log

def count_topn_occurances(recommendations,return_recommendations_df=False,N=5):
    """
    Input:
    ------
            recommendations:            dataframe with the topn recommendations
            retrutn_recommedations_df:  if True adds topn_to_str column to recommendations
                                        dataframe and returns it
            N:                          number of recommendations in topN
    Purpose:
    --------
            calculate how many times each topn was recommended to a customer,
            unique topn sets
    Return:
    -------
            dataframe with explanatory info for each topn        
            
    """
    #check is 'topn_to_str' column already exist if not add it 
    if  not('topn_to_str' in recommendations.columns.values):
    
        #convert list like topn to str 
        recommendations['topn_to_str']=recommendations['Top{}'.format(N)].apply(lambda x: ''.join(x))

    # count how many times each TopN was recommended
    topn_count=recommendations['topn_to_str'].value_counts().reset_index().rename(columns={'topn_to_str':'Times_recommended',       'index':'topn_to_str'})
    print('Unique Topn: {}'.format(topn_count.shape[0]))
    #keep unique TopN
    unique_topn = recommendations.drop_duplicates('topn_to_str')

    #get topN as list 
    topn_count = unique_topn[['Top{}'.format(N),'topn_to_str']].merge(right=topn_count,how='right',on='topn_to_str')
    topn_count=topn_count.drop('topn_to_str',axis=1)

    #seperate topn products
    for n in range(1,N+1):
        topn_count['top{}'.format(n)]=topn_count['Top{}'.format(N)].apply(lambda x: x[n-1])
 
    
    if return_recommendations_df==True:
        return topn_count,recommendations
    else:    
        return topn_count

def count_product_in_topn(product_name,recommendations):
    """
    Input:
    -----
        product_name:   product name as to calculate the occurances in topn 
        recommendatios: dataframe with all the recommendatios 
    Purpose:
    -------
            counts product's occurances in all the topn recommendations
    Output:
    ------  pass

    """
    #check is 'topn_to_str' column already exist if not add it 
    if  not('topn_to_str' in recommendations.columns.values):
    
        #convert list like topn to str 
        recommendations['topn_to_str']=recommendations['Top5'].apply(lambda x: ''.join(x))

    #check if product is in topn as str mode
    return recommendations['topn_to_str'].apply(lambda x: product_name in x).value_counts()

def product_popularity_in_recommendations(recommendation_df,N):
    """
    Input:
    ------
            recommendation_df: dataframe with topN for each customer
            N:                 number of recommendations
    Purpose:
    --------
            conducts after recommendation evaluation: 
            1.counts all the unique prodicts in all TopN recommendations
            2.calculates product frequency 
    Output:
    -------
            list with ['ProductName','frequency']               
    """
    all_the_recommendations=[]
    recommendations=recommendation_df['Top{}'.format(N)]
    for recommendation_list in recommendations:
        for product in recommendation_list:
            #create a list with all the recommendations for every customer
            all_the_recommendations.append(product)
            
    #product frequnecy in recommendations as a list 
    recommendation_frequency=np.unique(all_the_recommendations,return_counts=True)
    #count the different products in all the topN recommendation 
    unique_products=len(np.unique(all_the_recommendations))
    #create dictionary
    frequency_dict=dict(zip(np.unique(all_the_recommendations,return_counts=True)[0],np.unique(all_the_recommendations,return_counts=True)[1]))
    #descending order
    frequency_dict={k: v for k, v in sorted(frequency_dict.items(), key=lambda item: item[1],reverse=True)}
     
    print('Unique products in all TopN: {} '.format(unique_products))  
    return frequency_dict 

def hit_popularity_matrix(hit_products):
    """
    Input:
    -----
        hit_products: list with all the products in hits
    Purpose:
    --------
            calculate product popularity in all the hits 
    Output:
    ------
            dataframe preview Product| Number_of_hits |           
    """
    #value counts 
    products , counts = np.unique(np.array(hit_products) , return_counts=True)
    #create dataframe
    hit_popularity=pd.DataFrame(columns=['Product','Hits'])
    hit_popularity['Product']=products
    hit_popularity['Hits']=counts
    #descending order by Hits
    hit_popularity=hit_popularity.sort_values('Hits',ascending=False).set_index(np.arange(1,hit_popularity.shape[0]+1))
    return hit_popularity
    
#------------------------------ LEAVE ONE OUT -----------------------------------------------------    
def Leave_highest_rating_out(dataframe):
    """
    Input:
    ------
            dataframe : dataframe in RS format
    Purpose:
    -------
            removes from the train dataframe each users highest rated product 
    Output:
    ------
            train dataframe after the removing , dataframe with the removed rows                
    """
    rating_column=dataframe.columns[-1]
    print('Rows before: {} \n'.format(dataframe.shape[0]))
    dataframe=dataframe[['CustomerId','ProductId',rating_column]]
    dataframe=dataframe.sort_values(['CustomerId',rating_column],ascending=False)

    #to_train index for each customer's highest rated product 
    highest_rated_index=dataframe.groupby('CustomerId').apply(
                                                       lambda first_rating: first_rating.index[0]).reset_index()[0].values

    highest_rated_products=dataframe.loc[highest_rated_index]
    # drop them 
    dataframe.drop(index=highest_rated_index,inplace=True)
    print('Rows after: {} \n'.format(dataframe.shape[0]))

    return dataframe,highest_rated_products,rating_column 

def LeaveOneOut_hitrate(left_out_products,recommendations,N):
    """
    Input:
    -----   left_out_products: removed products 
            recommendations:   dataframe with the topn recommendations
            N:                 number of recommendations
    Purpose:
    --------
            count in how many cases the removed Product is in the TopN
    Output:
    -------
            hit_rate, number of hits, customers in topn, time passed           
    """
    begin=t.time()
    in_topn=0
    customerid_in_topn=[]
    for customer in recommendations.index:
        topn = recommendations.loc[customer]['Top{0}_Ids'.format(N)]
        leaved_out= left_out_products.loc[left_out_products['CustomerId'] == customer,['ProductId']].values[0]
        #print(topn, leaved_out)

        if leaved_out in topn:
            in_topn += 1
            customerid_in_topn.append(customer)
    in_topn_rate = round(in_topn / recommendations.shape[0],2)
    time_passed=t.time() - begin 

    return in_topn_rate , in_topn , customerid_in_topn ,time_passed

def leave_one_out_info(customer,data,removed_ratings,products_df,customers_df,rating_column):
    """
    Input:
    ------
            customer:        customer id 
            data:            train dataframe before LOO (train)
            removed_ratings: dataframe with removed LOO products 
            prodcucts_df:    dataframe with all the product ids and names (for id_to_name())
            customers_df:    dataframe with all the customer ids and names (for id_to_name())
            rating_column:   specifies the rating type
    Purpose:
    --------
            Find highest rating and highlight it 
            (Varification for Leave_Highest_rating_out)
            
    Output:
    ------- 
            Dataframe with all the rated products and the highest rated product highlighted
            (double entry for highest rating means correct LOO process)            
    """
    #find all products the customer rated
    customer_df=data.loc[data['CustomerId']== customer]
    #highest rated products per customer which where removed for Loo_rate
    highest_rated=removed_ratings.loc[removed_ratings['CustomerId']== customer]
    #add removes product
    customer_df=pd.concat([customer_df,highest_rated])
    #get names from ids
    customer_df['Customer_Name']=id_to_name(sample=[customer],products_df=products_df,
                                            customers_df=customers_df,choose='customers')[0]
    customer_df['Product_Names']= id_to_name(sample=customer_df['ProductId'].values,products_df=products_df,
                                            customers_df=customers_df,choose='products')
    #all customer products(+removed)                                       
    customer_df=customer_df[['Customer_Name','Product_Names',rating_column]]
    customer_df=customer_df.reset_index()
    
    #return dataframe with highlighted highest rate 
    return customer_df.style.apply(lambda x: ['background:yellow' if x == customer_df[rating_column].max() else 
                                            'background:white' for x in customer_df[rating_column]])    
 
# -------------------------------------------------------RS MODEL 2--------------------------------------------------------------------

# ------------------------------------------------------GET PREDICTIONS--------------------------------------------------------------------

 
def get_predictions_2(customer_id,train,model,customer_dict,product_dict,products_df,customers_df,N=5):
    """
    Input:
    -----
            customer_id: Customer Id (for each customer in the dataset)
            train:       training dataset, not in csr matrix format
            model:       trainned model 
            customer_dict: customer ids in LightFM format
            product_dict:  product ids in LightFM format
            products_df:   all available products dataframe
            customers_df:  all available customers dataframe
            N:             number of recommendations in top N, default: 5
    Purpose:
    --------  
            For given model customerId (int) predicts rating for unbought products
    Output:
    ------ 
            List with prediction for provided product list.    
    """
    # customer_id=int(customer_id)
    already_bought=train.columns[train.iloc[customer_id] > 0 ].values
    #make predicitons for products that are not bought
    to_predict=dict((key, product_dict[key] ) for key in product_dict if key not in already_bought)
    predictions=model.predict(user_ids=int(customer_id),item_ids=list(to_predict.values()),item_features=None,user_features=None,num_threads=2)
    
    #topn
    scores=pd.DataFrame(index=to_predict.values(),data=predictions).rename(columns={0:'ratings'})
    scores=scores.sort_values('ratings',ascending=False)
    # get topn
    scores=scores.head(N)
#     print(scores)
    
    #create reverse dict 
    reverse_product_dict= {value: key for key, value in product_dict.items()}

    product_ids=[]
    product_names=[]
    for model_id in scores.index:
        product_id=reverse_product_dict[model_id]
        product_ids.append(product_id)
        product_names.append(id_to_name(sample=[product_id], products_df=products_df, customers_df=customers_df, choose='products')[0])

    return  customer_id , product_ids ,product_names

def topn_recommedation(train,model,customer_dict,product_dict,products_df,customers_df,N=5):
    """
    Input:
    ------
            train:         train data in uncompressed matrix form 
            customer_dict: customers to make predictions (whole dictionary ,segmented dictionary),
                           in LightFM format
            product_dict:  products in LighFM format
            products_df:   all products available
            customers_df:  all customers
            N:             number of recommendations in Top N
    Purpose:
    --------
            transform model predictions to dataframe 

    Output:
    ------- 
            dataframe with customers' topn recommendations   
    """

    
    customerids=[]
    customer_names=[]
    top_products=[]
    top_product_ids=[]
    for customer_id , customer_model_id  in customer_dict.items():

        (customer_modelId , topn_ids , topn_prod) =get_predictions_2(customer_id=customer_model_id,train=train,model=model,customer_dict=customer_dict,product_dict=product_dict,products_df=products_df,customers_df=customers_df)
        customerids.append(customer_id)
        
        #get customer name
        customer_name=id_to_name(sample=[customer_id], products_df=products_df, customers_df=customers_df, choose='customers')[0]
        customer_names.append(customer_name)
        top_products.append(topn_prod)
        top_product_ids.append(topn_ids)

    #create df
    recommendations=pd.DataFrame(index=customerids)
    recommendations['CustomerName']=customer_names
    recommendations['Top{}'.format(N)]=top_products
    recommendations['Top{}_Ids'.format(N)]=top_product_ids
    return recommendations    
     
 
#-------------------------------- HYPERPARAMETER TUNNING-----------------------------------------

def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            #"loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "loss":np.random.choice(["warp", "bpr"]), 
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 50),
        }


def random_search(train, test, num_samples=10, num_threads=1):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    on the test set.

    Parameters
    ----------

    train:       np.float32 coo_matrix of shape [n_users, n_items]
                 Training data.
    test:        np.float32 coo_matrix of shape [n_users, n_items]
                 Test data.
    num_samples: int, optional
                 Number of hyperparameter choices to evaluate.


    Returns
    -------

    generator of (tunning scores, hyperparameter dict, fitted model)

    """
    
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)

        #auc 
        train_auc_score = auc_score(model, train,  num_threads=num_threads).mean()
        test_auc_score = auc_score(model, test,   num_threads=num_threads).mean()

        #precision
        train_precision = precision_at_k(model,train,k=5).mean()
        test_precision = precision_at_k(model,test,k=5).mean()

        #recall
        train_recall = recall_at_k(model,train,k=5).mean()
        test_recall = recall_at_k(model,test,k=5).mean()

        hyperparams["num_epochs"] = num_epochs

        yield (train_auc_score, test_auc_score, train_precision, test_precision, train_recall, test_recall, hyperparams, model)

def hyperparameter_tuning(trainset, testset, tuning_metric='Precision', num_samples=10):
    """
    Input:
    ------
           trainset:      train data in compressed matrix form. 
           testset:       test data in compressed matrix form. 
                          (used here for evaluating purpose not training)
           tuning_metric: metric according to which the tuning is performed.
                          Precision, Recall, AUC Score
           num_samples:   number of iteration for the random search.
                       

    Purpose:
    --------
            Performs hyperparameter tuning and returns fitted model to the best tuning metric 
            score hyperparameters 

    Output:
    -------
           train_auc_score, test_auc_score, train_precision, test_precision, train_recall, test_recall,
           hyperparams, model, num_sampes, tuning_time, tuning_metric 
    """
    #store time 
    tuning_timing=t.time()

    #get tuning metric
    if tuning_metric == 'Precision':
        tuning_number = 4
    elif tuning_metric == 'Recall':
        tuning_number = 6
    elif tuning_metric == 'AUC':
        tuning_number = 2
    #tuning
    (train_auc_score, test_auc_score, train_precision, test_precision, train_recall, test_recall, hyperparams, model) = max(random_search(train=trainset,
                                                                test=testset, num_samples=num_samples), key=lambda x: x[tuning_number])


    tuning_timing= t.time() - tuning_timing

    return train_auc_score, test_auc_score, train_precision, test_precision, train_recall, test_recall, hyperparams, model, num_samples, tuning_timing, tuning_metric 
 
def training_log2(rating_column,algo,tuning_time='-',num_samples=0, tuning_score='-',tuning_metric='Precision', create_log=False,hyperparams=dict(),
                  LOO=False):
  """
      Input:
      ------
            rating_column:   specifies the rating type used 
                             (e.g.: Quantity, Frequency1, Frequency2, New_Frequency)
            algo:            algorithm      
            tuning_time:     time elapsed for randomized search, 
                             if value equals to '-' no hyperparameter tuning has occured.
            num_samples:     number of samples to perform random search
            tuning_metric:   metric that used during hyperparameter tuning
                             (e.g.: Presicion, Recall, AUC Score)
            tuning_score:    tuning metric score produced from hyperparameter tuning
            create_log:      If True, creates new log, if False, adds row to existent
            hyperparameters: dictionary with best fitted hyperparameters from tuning
            LOO:             Boolean varieable indicates if Leave One Out (Data) method is used.

      Purpose:      
      -------
            creates dataframe with log-like purpose (training history)
      Output:
      -------
            dataframe with aggregatted information regarding training trails
  """
  #create log
  if create_log == True :
    #create dataframe and store it to disk
    cols=['Date', 'Data', 'Algo', 'Tunning', 'Tunning_time', 'Tuning_Samples', 'Parameters', 'Tuning_Metric', 'Tuning_score', 'Leave_one_out']
    log=pd.DataFrame(columns=cols)
    dill.dump(log,open('train_log2.d','wb'))  
  else:
    with open('train_log2.d','rb') as f:
      log=dill.load(f)

  if tuning_time == '-':
    tuning_flag = False
    num_samples= '-'
    tuning_score='-'
    tuning_metric='-'
  else:
    tuning_flag = True
    tuning_time=round(tuning_time,2)
    tuning_score=round(tuning_score,2)
   
      
  #create list to add to the DataFrame 
  update_log=[datetime.now(),rating_column, str(algo).split()[0][17:], tuning_flag, tuning_time,num_samples,
             hyperparams, tuning_metric, tuning_score, LOO]
  
  #add list
  log=log.T
  log[log.shape[1] + 1] = update_log
  log=log.T
 
  #update dill file 
  dill.dump(log,open('train_log2.d','wb'))
  
  return log     

 
 
