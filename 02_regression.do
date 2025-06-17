/*This file takes the data on the UN and control variables from final_data and runs regression specification to figure out a decent model. The top model uses logs of the dependent variable*/

use final_data, clear


replace battle_fat=battle_fat/1000
collect clear

xtset country_num year

collect: xtreg share battle_fat TerritoryOccupied human_rights_score western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc i.year if Israel_dummy==0, re cluster(country_num)


collect: xtreg share battle_fat TerritoryOccupied human_rights_score western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc i.year if Israel_dummy==0, fe cluster(country_num)


collect: xtreg share battle_fat TerritoryOccupied human_rights_score western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc if Israel_dummy==0, be

collect: xtreg share battle_fat TerritoryOccupied human_rights_score western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc i.year if Israel_dummy==0&share>0&share<1, re cluster(country_num)


collect: regress share  battle_fat human_rights_score if Israel_dummy==1, robust



collect label levels colname share "UNGA Critical Resolutions Share" battle_fat "War Fatalities (1000s)" TerritoryOccupied "Occupying Country Indicator" human_rights_score "Human Rights Score" western_affiliate "US Military Ally" PermanentUNSCMember "Permanent Security Council Member" log_population "Population (Logs)" log_gdppc "GDP per capita (Log of current $)" allyXconflict "US Ally and Conflict Interaction" western_laggard "US Ally with Poor Human Rights Record" NuclearPowerOutsideNPT "Nuclear Weapons Outside of Treaty" , modify



collect style autolevels result _r_b _r_se
collect style autolevels colname share battle_fat TerritoryOccupied human_rights_score western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc 

collect layout (colname#result result[N r2]) (cmdset)
collect stars _r_p 0.01 "***" 0.05 "**" 0.1 "*", attach(_r_b) /*add stars*/
collect style header result, level(hide) /*Get rid of coefficient and standard error row names.*/
collect style cell result[_r_se], sformat((%s)) /*Put standard errors in parens*/
collect label levels cmdset 1 "Random Effects (ex. Israel)" 2 "Fixed Effects (ex. Israel)" 3 "Between Effects (ex. Israel)" 4 "Random Effects (Positive Share)" 5 "Israel Only" /*Labeling the columns*/

collect style cell, nformat(%8.4f) /*Only displaying two decimal points*/
collect style header result[N r2], level(label)
collect label levels result N "Observations" r2 "R-squared", modify
collect title Table 2: Panel Regression Results
collect notes "Panel data regression coefficient estimates, dependent variable is the share of annual critical UNGA resolutions that are directed at each country. Standard errors robust for correlation at the country level in parentheses. ***, **, * denote 99%, 95%, and 90% statistical significance, respectively.  Data exclude Israel except for the final column which includes only Israel. The independent variables include the Uppsala Conflict Data Program's measure of total fatalities associated with those conflict involving that particular country, an indicator variable for whether the country is occupying other territory, as measured by the Geneva Institute's Rule of Law and Armed Conflict Database, an indicator variable for whether a country is a US military ally (proxied by NATO or Major Non-NATO Ally status), the country's annual overall human rights score, as measured by the CIRights project, and an indicator variable for whether a country is a nuclear power outside the NPT treaty (as measured by SIPRI's Annual Yearbook). Columns 1, 2, and 3, present random effects, fixed effects, and between effects panel estimates of equation (1), respectively. Column 4 excludes countries subject to no criticism in a given year to mitigate the zero inflation problem. Column 5 presents regression including only Israeli observations. Coefficients for variables fixed for Israel (such as its nuclear power status or Occupying territory status) are therefore undefined." 

collect style cell result[N], nformat(%7.0f)
collect style cell result[r2], nformat(%7.3f)

collect export table2.docx, replace


sum share if PermanentUNSCMember==1, detail

sort year
list share year battle_fat if Israel_dummy==1

list share year if country=="South Africa"

sum human_rights_score if Israel_dummy==1, detail


