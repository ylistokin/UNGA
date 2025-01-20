/*This file takes the data on the UN and control variables from final_data and runs regression specification to figure out a decent model. The top model uses logs of the dependent variable*/

use final_data, clear

collect clear

collect: regress log_count log_fatal TerritoryOccupied log_hum_rights western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc if Israel_dummy==0, cluster(country)

collect: regress log_count log_fatal TerritoryOccupied log_hum_rights  western_affiliate NuclearPowerOutsideNPT western_laggard PermanentUNSCMember log_population log_gdppc if Israel_dummy==0, cluster(country)

collect: regress log_count log_fatal TerritoryOccupied log_hum_rights western_affiliate NuclearPowerOutsideNPT western_laggard PermanentUNSCMember log_population log_gdppc year_dummy* region_dummy* if Israel_dummy==0, cluster(country)

collect: regress count log_fatal TerritoryOccupied log_hum_rights western_affiliate western_laggard NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc if Israel_dummy==0, cluster(country)

collect label levels colname count "UNGA Resolutions" log_fatal "War Fatalities (Logs)" TerritoryOccupied "Occupying Country Indicator" log_hum_rights "Human Rights Score (Logs)" western_affiliate "US Military Ally" PermanentUNSCMember "Security Council Member" log_population "Population (Logs)" log_gdppc "GDP per capita (Log of current $)" allyXconflict "US Ally and Conflict Interaction" western_laggard "US Ally with Poor Human Rights Record" NuclearPowerOutsideNPT "Nuclear Weapons Outside of Treaty" , modify



collect style autolevels result _r_b _r_se
collect style autolevels colname log_count log_fatal TerritoryOccupied log_hum_rights western_affiliate NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc western_laggard 

collect layout (colname#result result[N r2]) (cmdset)
collect stars _r_p 0.01 "***" 0.05 "**" 0.1 "*", attach(_r_b) /*add stars*/
collect style header result, level(hide) /*Get rid of coefficient and standard error row names.*/
collect style cell result[_r_se], sformat((%s)) /*Put standard errors in parens*/
collect label levels cmdset 1 "Specification 1: Base" 2 "Specification 2: Interaction Terms" 3 "Specification 3: Interactions and Time/Region Fixed Effects" 4 "Specification 4: Dependent Variable in Levels" /*Labeling the columns*/

collect style cell, nformat(%8.3f) /*Only displaying two decimal points*/
collect style header result[N r2], level(label)
collect label levels result N "Observations" r2 "R-squared", modify
collect title Table 2: Regression Results
collect notes "Regression coefficient estimates, dependent variable is the log of annual critical UNGA resolutions about each country, standard errors in parentheses. ***, **, * denote 99%, 95%, and 90% statistical significance, respectively.  Data exclude Israel. Standard errors clustered at the country level. The dependent variable in all regressions is the number of UNGA resolutions about a country in a year. The independent variables include the Uppsala Conflict Data Program's measure of total conflict intensity for all conflicts involving the country in the given year and the total fatalities associated with those conflicts, an indicator variable for whether the country is occupying other territory, as measured by the Geneva Institute's Rule of Law and Armed Conflict Database, an indicator variable for whether a country is a US military ally (proxied by NATO or Major Non-NATO Ally status), the country's annual overall human rights score, as measured by the CIRights project, and an indicator variable for whether a country is a nuclear power outside the NPT treaty (as measured by SIPRI's Annual Yearbook). Specifications 2-3 expand the specification by adding interaction terms and region and year effects. In Specification 4, the dependent variable is in levels rather than logs. "


collect style cell result[N], nformat(%7.0f)
collect style cell result[r2], nformat(%7.3f)

collect export table2.docx, replace



regress count log_fatal TerritoryOccupied log_hum_rights western_affiliate western_laggard NuclearPowerOutsideNPT PermanentUNSCMember log_population log_gdppc if Israel_dummy==0&count>=1, cluster(country)
