/*This file takes the data on the UN and control variables from final_data and runs the Oaxaca decomposition*/

use final_data, clear


collect clear


collect: oaxaca count log_fatal TerritoryOccupied log_hum_rights western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc, by(Israel_dummy) weight(1) cluster(country) relax

collect: oaxaca count log_fatal TerritoryOccupied log_hum_rights western_affiliate NuclearPowerOutsideNPT allyXconflict PermanentUNSCMember log_population log_gdppc, by(Israel_dummy) weight(1) cluster(country) relax

collect: oaxaca count log_fatal TerritoryOccupied log_hum_rights western_affiliate NuclearPowerOutsideNPT allyXconflict PermanentUNSCMember log_population log_gdppc year_dummy1-year_dummy35 region_dummy18, by(Israel_dummy) weight(1) cluster(country) relax




collect label levels cmdset 1 "Specification 1" 2 "Specification 2" 3 "Specification 3" , modify
collect label levels colname difference "Total Difference" explained "Explained Difference" unexplained "Unexplained Difference", modify
collect label levels result _r_b "Estimate", modify

collect style cell, nformat(%8.2f)


collect style header coleq[overall], level(hide)
collect stars _r_p 0.01 "***" 0.05 "**" 0.1 "*", attach(_r_b)
collect style cell result[_r_se], sformat((%s)) /*Put standard errors in parens*/ 

collect layout (cmdset#coleq[overall]#result[_r_b _r_se]) (colname[difference explained unexplained])

collect title Table 3: Blinder-Oaxaca Decomposition Results
collect notes "Blinder-Oaxaca decomposition estimates and standard errors, dependent variable is the number of annual critical UNGA resolutions about each country, standard errors in parentheses. ***, **, * denote 99%, 95%, and 90% statistical significance, respectively. Total difference is the difference between the average (over the years 1981-2023) number of annual critical UNGA resolutions about Israel and the average for the other 193 countries in the sample. Explained difference is the predicted difference between the average for Israel versus the other countries, given the control variables. Unexplained difference is what remains after the explained difference is substracted from the overall difference. Standard errors clustered at the country level." 



collect export table3.docx, replace
