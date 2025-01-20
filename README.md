# UNGA
Data Availability Statement:

All files discussed in this data availability statement are available at https://github.com/ylistokin/UNGA (this GitHub repository). 
 
The primary input databases for the paper are:
1.	The text of each UNGA resolution from 1981 to 1994 from Mesquita, Rafael; Pires, Antonio, 2024, "UN General Assembly Resolutions Text". 
2.	Various resolutions data from 1995-2023 from United Nations General Assembly Resolutions: Voting Data and Issue Categories, by Voelsen, Daniel; Bochtler, Paul; Majewski, Rebecca. In addition to the supplementary data, the full text of each UNGA resolution in this period was obtained directly from these authors.  
3.	A count of the annual number of critical UNGA resolutions about each country. (GPT_resolution_counts.csv.)
4.	CI Rights Dataset (cirights v2.8.27.23_public.xlsx)
5.	UCDP Battle-Related Deaths Dataset version 24.1 (dyadic version) (BattleDeaths_v24_1.xlsx)
6.	UCDP Dyadic Dataset version 24.1 (Dyadic_v24_1.xlsx)
7.	The World Bank Databank (for economic and demographic variables)  (P_Data_Extract_From_World_Development_Indicators.xlsx)
8.	OECD membership, NATO Membership, US Major non-Nato Allies. Data on Occupation was obtained from the RULAC (Rule of Law in Armed Conflict) map program, using the “Military Occupation” filter (Geopol_1981_2023.xlsx).
9.	UN Watch database of critical resolutions (UN Watch GA Resolutions 2015-August 2024.xlsx)

 #1/#10 is analyzed by Chat GPT in the file GPT_Voelsen.ipynb. #2 is analyzed by ChatGPT in the file GPT_on_Mesquita.ipynb (Note that full replication of the results requires a GPT APE Key). . The results are turned into data on the annual number of country year resolutions in Annual_Country_Resolutions_MV.ipynb. The countryXyear results (database #3) can be found in GPT_resolution_counts.csv. 
Figure 1 from the paper is produced by F1_score.ipynb.  

The stata do file 01_make_variables_8123.do combines databases #3 through #9, refines variables, with the output saved as Stata file final_data.dta. The unit of observations in this dataset is CountryxYear
Summary statistics (Table 1) is produced by 02_sumstat.do. Regression coefficients (Table 2) is produced by 02_regression.do. The Oaxaca decomposition (Table 3) is 

