*This file is the roadmap of how I examine the data. 01_make_variables_81_23 takes in all the existing downloaded datasets (Voelsen and Mesquita analyzed by ChatGPT on python, CI Human Rights, and Uppsala, World Bank,) and creates the final dataset. 02_sumstat creates the table of summary statistics, 02_regression then creates a table with a regression model, excluding Israel. 02_oaxaca then runs the Oaxaca decomposition 

do 01_make_variables_81_23
do 02_sumstat
do 02_regression (or 02_regression_levels if I am showing levels rather than logs. )
do 02_oaxaca