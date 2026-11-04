"""This script will be run in LEAP to add Centralized Generation & Storage Processes to the Moldova model"""

#imports 
import pandas as pd
import win32com 
import win32com.client

LEAP = win32com.client.Dispatch("LEAP.LEAPApplication")
print(LEAP.ActiveArea.Name)   # should print moldova_eraa_v1

#setting parameters
file_path= r"C:\Users\ehina\OneDrive - SEI\GA GUMA - WP 2 Electricity Resource Adequacy Assessment\Modeling\Data\From SEI US\Centralized generation and storage processes\LEAP_centralized_generation_storage_processes.xlsx"
sheet_name= "Processes"

#creating new dataframe from the Excel file with JUST processes and their fuels 
df = pd.read_excel(file_path, sheet_name=sheet_name, header=3, usecols= "D,K")
df = df.dropna(how="all")
print(df)


#adding processes to Process Branch under Centralized Generation and Storage Branch, which is under Transformation
Parent= r"Transformation\Centralized Generation and Storage\Processes"
for i in range(len(df)): 
    Name = df.iloc[i, 0]
    BranchType = "Process"
    Properties = "Fuel=" + str(df.iloc[i, 1])
    LEAP.AddBranch(301, Name, BranchType, Properties)


#this was a test in LEAP 
