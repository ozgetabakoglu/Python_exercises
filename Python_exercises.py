
##################################################
# List Comprehensions
##################################################

# ###############################################
# # TASK 1: Using the List Comprehension structure, capitalize the names of the numeric variables in the car_crashes data and add NUM to the beginning.
# ###############################################
#
# # Expected Output
#
# # ['NUM_TOTAL',
# #  'NUM_SPEEDING',
# #  'NUM_ALCOHOL',
# #  'NUM_NOT_DISTRACTED',
# #  'NUM_NO_PREVIOUS',
# #  'NUM_INS_PREMIUM',
# #  'NUM_INS_LOSSES',
# #  'ABBREV']
#

import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("car_crashes")
df.columns
df.info()


["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]


# ###############################################
# # TASK 2: Using the List Comprehension structure, write "FLAG" after the names of the variables in the car_crashes data that do not contain "number" in their names.
# ###############################################
#
## Notes:
# # All variable names must be uppercase.
# # A single list should be made with comp.
#
# # Expected output:
#
# # ['TOTAL_FLAG',
# #  'SPEEDING_FLAG',
# #  'ALCOHOL_FLAG',
# #  'NOT_DISTRACTED',
# #  'NO_PREVIOUS',
# #  'INS_PREMIUM_FLAG',
# #  'INS_LOSSES_FLAG',
# #  'ABBREV_FLAG']


[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]

# #############################################
# # Task 3: Using the List Comprehension structure, select the names of the variables that are DIFFERENT from the variable names given below and create a new dataframe.
# #############################################
#
og_list = ["abbrev", "no_previous"]
#
## Notes:
# # First, create a new list named new_cols using list comprehension according to the list above.
# # Then create a new df by selecting these variables with df[new_cols] and name it new_df.
#
# # Expected output:
#
# #    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# # 0 18.800     7.332    5.640          18.048      784.550     145.080
# # 1 18.100     7.421    4.525          16.290     1053.480     133.930
# # 2 18.600     6.510    5.208          15.624      899.470     110.350
# # 3 22.400     4.032    5.824          21.056      827.340     142.390
# # 4 12.000     4.200    3.360          10.920      878.410     165.630
#

og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df.head()





##################################################
# Pandas Exercises
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Task 1: Identify the Titanic dataset from the Seaborn library.
#########################################
df = sns.load_dataset("titanic")
df.head()
df.shape

#########################################
# Task 2: Find the number of male and female passengers in the Titanic dataset described above.
#########################################

df["sex"].value_counts()


#########################################
# Task 3: Find the number of unique values for each column.
#########################################

df.nunique()

#########################################
# Task 4: Find the unique values of the variable pclass.
#########################################

df["pclass"].unique()


#########################################
# Task 5: Find the number of unique values of pclass and parch variables.
#########################################

df[["pclass","parch"]].nunique()

#########################################
# Task 6: Check the type of the embarked variable. Change its type to category. Check the repetition type.
#########################################

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()


#########################################
# Task 7: Show all the sages of those with embarked value C.
#########################################

df[df["embarked"]=="C"].head(10)


#########################################
# Task 8: Show all the sages of those whose embarked value is not S.
#########################################

df[df["embarked"] != "S"]["embarked"].unique()

df[~(df["embarked"] == "S")]["embarked"].unique()



#########################################
# Task 9: Show all the information for female passengers younger than 30 years old.
#########################################

df[(df["age"]<30) & (df["sex"]=="female")].head()


#########################################
# Task 10: Show the information of passengers whose Fare is over 500 or 70 years old.
#########################################

df[(df["mouse"] > 500 ) | (df["age"] > 70 )].head()


#########################################
# Task 11: Find the sum of the null values in each variable.
#########################################

df.isnull().sum()


#########################################
# Task 12: drop the who variable from the dataframe.
#########################################

df.drop("who", axis=1, inplace=True)


#########################################
# Task 13: Fill the empty values in the deck variable with the most repeated value (mode) of the deck variable.
#########################################


type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()


#########################################
# Task 14: Fill the empty values in the age variable with the median of the age variable.
#########################################

df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()

#########################################
# Task 15: Find the sum, count, mean values of the Pclass and Gender variables of the survived variable.
#########################################

df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})


#########################################
# Task 16: Write a function that returns 1 for those under 30, 0 for those above or equal to 30.
# Create a variable named age_flag in the titanic data set using the function you wrote. (use apply and lambda constructs)
#########################################

def age_30(age):
     if age<30:
         return 1
     else:
         return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))


df["age_flag"] = df["age"].apply(lambda x: 1 if x<30 else 0)


#########################################
# Task 17: Define the Tips dataset from the Seaborn library.
#########################################

df = sns.load_dataset("tips")
df.head()
df.shape


#########################################
# Task 18: Find the sum, min, max and average of the total_bill values ​​according to the categories (Dinner, Lunch) of the Time variable.
#########################################

df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})

#########################################
# Task 19: Find the sum, min, max and average of total_bill values ​​by days and time.
#########################################

df.groupby(["day","time"]).agg({"total_bill": ["sum","min","mean","max"]})

#########################################
# Task 20: Find the sum, min, max and average of the total_bill and type values ​​of the female customers, according to the day of the lunch time.
#########################################


df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "min","max","mean"],
                                                                           "type": ["sum","min","max","mean"],
                                                                            "Lunch" : lambda x: x.nunqiue()})


#########################################
# Task 21: What is the average of orders with size less than 3 and total_bill greater than 10?
#########################################


df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean() # 17.184965034965035



#########################################
# Task 22: Create a new variable called total_bill_tip_sum. Let him give the sum of the total bill and tip paid by each customer.
#########################################
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()



#########################################
# Task 23: Sort the total_bill_tip_sum variable from largest to smallest and assign the first 30 people to a new dataframe.
#########################################

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape



