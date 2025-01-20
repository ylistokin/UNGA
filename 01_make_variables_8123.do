/*This file creates the primary dataset. There are a few critical inputs. 
1.  C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\GPT_Counts\GPT_resolution_counts.xlsx, which has a country-year unit of observation and includes counts of the total number of critical UNGA resolutions for the country-year from ChatGPT applied to the Voelsen data (1994-2023) and Mesquita Data (1981-1993). (This is all accomplished in Google Colab, with details available at my google drive,  "/content/gdrive/MyDrive/UNGA/Roadmap.ipynb")
2. Human rights data from cirights C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\RI Human Rights\v2.8.27.23_public.xlsx.
3. This data is merged with World Bank economic and demographic data from   and the Voelsen data updated_merge_princeton_and_counts_wbdata_with_mnna.xlsx, , lots of control variables from the World Bank data at C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\P_Data_Extract_From_World_Development_Indicators.xlsx other controls, 
4.  1 Uppsalla conflict data from C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\UPCD\UcdpPrioConflict_v24_1.csv" and 
*/


**************************************Input Condemnation Counts Data, https://cirights.com************************/
import delimited "C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\GPT_Counts\GPT_resolution_counts.csv", varnames(1) clear

replace country = trim(country)
drop if year<1981
sort country year


/*Fixing up country names to correspond to human rights data*/
replace country="Bosnia Herzegovenia" if country=="Bosnia and Herzegovina"
replace country="Cote d'Ivoire" if country=="Côte d'Ivoire"
replace country="Korea, Democratic People's Republic of" if country=="Democratic People's Republic of Korea"
replace country="Korea, Democratic People's Republic of" if country=="Democratic People's Republic of Korea"
replace country="Congo, Democratic Republic of" if country=="Democratic Republic of the Congo"
replace country="Estonia" if country=="Estonia and Latvia"
replace country="Libya" if country=="Libyan Arab Jamahiriya"
replace country="United States of America" if country=="United States"
replace country="Congo, Democratic Republic of" if country=="Zaire"&year<1988



save temp, replace





/**************************************Input Rhode Island Human Rights data, https://cirights.com*/
import excel "C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\RI Human Rights\cirights v2.8.27.23_public.xlsx", firstrow clear

replace country = trim(country)
drop if year<1981
sort country year

/*The dataset is missing 2022 and 2023, duplicating 2021 to add those years in accordingly*/
gen duplicates = 3 if year==2021
expand duplicates
drop duplicates
sort country year
by country year: gen index=_n
replace year=2022 if index==2
replace year=2023 if index==3

/*Fixing country names*/
replace country="Myanmar" if country=="Burma"
replace country="Russia" if country=="Soviet Union"
replace country="Serbia" if country=="Serbia and Montenegro"
replace country="Serbia" if country=="Yugoslavia"
replace country="Yemen" if country=="Yemen Arab Republic"
replace country="Czech Republic" if country=="Czechoslovakia"


merge 1:1 country year using temp


*Dropping Palestine and Taiwan and one random misspelling
drop if country=="Taiwan"
drop if _merge==2 
drop _merge

sort country year
save temp, replace




/********************************************************** Load the World Bank Data**************************/
import excel "C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\P_Data_Extract_From_World_Development_Indicators.xlsx", firstrow clear

*Fixing names
rename CountryName country
rename Time year

*Getting rid of earlier years
drop if year<1981

*Getting rid of weird duplicate observations (no countries that would be part of final merge)
duplicates drop country year, force

*Prepping for merge
sort country year

*Fixing up country names for the merge
replace country="Bahamas" if country=="Bahamas, The"
replace country="Bolivia" if country=="Bolivia Plurinational State of"
replace country="Bosnia Herzegovenia" if country=="Bosnia and Herzegovina"
replace country="Brunei" if country=="Brunei Darussalam"

replace country="Cape Verde" if country=="Cabo Verde"
replace country="Congo, Democratic Republic of" if country=="Congo, Dem. Rep."
replace country="Congo, Republic of" if country=="Congo, Rep."
replace country="Cote d'Ivoire" if country=="Cote dIvoire"
replace country="Czech Republic" if country=="Czechia"
replace country="East Timor" if country=="Timor-Leste"
replace country="Gambia, The" if country=="Gambia"
replace country="Guinea-Bissau" if country=="Guinea Bissau"
replace country="Korea, Democratic People's Republic of" if country=="Korea, Dem. People's Rep."
replace country="Iran" if country=="Iran, Islamic Rep."
replace country="Korea, Republic of" if country=="Korea, Rep."
replace country="Kyrgyz Republic" if country=="Kyrgyzstan"
replace country="Laos" if country=="Lao PDR"
replace country="Macedonia" if country=="North Macedonia"
replace country="Moldova" if country=="Republic of Moldova"
replace country="Russia" if country=="Russian Federation"
replace country="Micronesia, Federated States of" if country=="Micronesia, Fed. Sts."
replace country="Serbia and Montenegro" if country=="Serbia"&year<2006
replace country="Serbia" if country=="Serbia"&year>=2006
replace country="Sao Tome and Principe" if country=="Sao Tomé and Principe"
replace country="Slovak Republic" if country=="Slovakia"
replace country="Syria" if country=="Syrian Arab Republic"
replace country="Tanzania" if country=="United Republic of Tanzania"
replace country="United Kingdom" if country=="United Kingdom of Great Britain"
replace country="United States of America" if country=="United States"
replace country="Venezuela" if country=="Venezuela, RB"
replace country="Egypt" if country=="Egypt, Arab Rep."
replace country="State of Palestine" if country=="West Bank and Gaza"
replace country="Yemen" if country=="Yemen, Rep."
replace country="Turkey" if country=="Turkiye"
replace country="Vietnam" if country=="Viet Nam"
replace country="Saint Vincent and the Grenadines" if country=="St. Vincent and the Grenadines"
replace country="Saint Kitts and Nevis" if country=="St. Kitts and Nevis"
replace country="Saint Lucia" if country=="St. Lucia"
replace country="Palestine" if country=="State of Palestine"
replace country="Serbia" if country=="Serbia and Montenegro"


replace country=trim(country)


merge 1:1 country year using temp


*Getting rid of extraneous observations, such as those that don't belong to countries
drop if _merge==1
drop _merge
sort country year

save counts_and_rights_and_demo, replace





************************* Load the Uppsala data (there are two datasets, one for conflicts, the other for battle deaths ****************************** 
import excel "C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\Dyadic_v24_1.xlsx", firstrow clear
sort conflict_id dyad_id year
save temp, replace


*Merging the conflict data with the combat deaths data
import excel "C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\BattleDeaths_v24_1.xlsx", firstrow clear
*destring conflict_id, replace
*destring year, replace
sort conflict_id dyad_id year

merge 1:1 conflict_id dyad_id year using temp, force
destring year, replace


*Getting rid of old observations
drop if year<1981

*Getting the name of each country in the conflict of the basis of the sides variables. 
gen comma_present_a = strpos(side_a, ",") > 0
gen comma_present_b= strpos(side_b, ",") > 0
replace side_a = substr(side_a, 1, strpos(side_a, ",") - 1) if comma_present_a==1
replace side_b = substr(side_b, 1, strpos(side_b, ",") - 1) if comma_present_b==1
gen countrya1=side_a
gen countryb1=side_b



*expanding so that we have a separate observation for each country listed in the conflict id data. The maximum number of total countries I have for a given conflict_id is 9, since I have capped the number of countries on any one side at 6. 
expand=2
*Generating a country variable that is different for each observations of a conflict_id that has multiple combatants.
bysort conflict_id dyad_id year: gen num_var=_n

*Running a loop to create a country variable that can be used to merge with the UN data
gen country=""
replace country=countrya1 if num_var==1
replace country=countryb1 if num_var==2





*Cleaning the data of both unnecessary spaces in variables, variables, years, and unnecessary observations that don't have a country.

drop if country==""
drop countrya*
drop countryb*

*Making a variable for Palestine
replace country="Government of Palestine" if country=="Hamas"|country=="Fatah"|country=="PFLP"


*Keeping only countries that are parts of governments because those are the only ones I can merge with other national data*
gen gov_dummy = strpos(country, "Government of") > 0
keep if gov_dummy==1

*Fixing country names for easier merging, getting rid of the descriptor "Government of"
replace country = subinstr(country, "Government of", "", .)
replace country = trim(country)


*Fixing up country names for better merging between datasets
replace country="Yemen" if country=="Yemen (North Yemen)"|country=="Democratic Republic of Yemen"
replace country="Russia" if country=="Russia (Soviet Union)"
replace country="Burma" if country=="Myanmar (Burma)"
replace country="Congo, Democratic Republic of" if country=="DR Congo (Zaire)"
replace country="Cote d'Ivoire" if country=="Ivory Coast"
replace country="Zimbabwe" if country=="Zimbabwe (Rhodesia)"
replace country="Cambodia" if country=="Cambodia (Kampuchea)"
replace country="Macedonia" if country=="North Macedonia"
replace country="Bosnia Herzegovenia" if country=="Bosnia-Herzegovina"
replace country="Serbia and Montenegro" if country=="Serbia (Yugoslavia)"&year<2006
replace country="Serbia" if country=="Serbia (Yugoslavia)"&year>=2006
replace country="Kyrgyz Republic" if country=="Kyrgyzstan"
replace country="Congo, Republic of" if country=="Congo"

replace country="Serbia" if country=="Serbia and Montenegro"
replace country="Myanmar" if country=="Burma"
replace country="Vietnam" if country=="Vietnam (North Vietnam)"

drop if country=="Palestine" 





/*Collapsing the data so that there is one observation per country year, facilitating merging with the UN data. Before I do this, I need to create bunch of dummy variable to retain information and also fix up dates*/
*making dummy vars

gen interstate=0
replace interstate=1 if type_of_conflict=="2"
gen intrastate=0
replace intrastate=1 if type_of_conflict=="3"|type_of_conflict=="4"
gen territory_dum=0
replace territory_dum=1 if incompatibility=="1"|incompatibility=="3"
*Fixing up string vars for collapse
gen battle_fat=real(bd_best)
destring intensity_level, replace
destring conflict_id, replace

*Fixing up dates
gen num_start_date=date(start_date, "YMD")
gen num_start_date2=date(start_date2, "YMD")
gen year_start=year(num_start_date)
gen year_start2=year(num_start_date2)
gen duration=year-year_start
gen new=0
replace new=1 if year==year_start
gen resurgence=0
replace resurgence=1 if year==year_start2

sort country year


collapse (sum) battle_fat total_intensity=intensity_level (max) conflict_id max_intensity=intensity_level new resurgence interstate territory_dum intrastate duration (count) total_conflicts=intensity_level, by(country year)
   
sort country year




merge 1:1 country year using counts_and_rights_and_demo

drop if _merge==1



*Now fixing the data up so that it has non missing variables with value of zero for multiple variables from the conflict and other datasets
replace total_intensity=0 if _merge==2
replace max_intensity=0 if _merge==2
replace new=0 if _merge==2 
replace resurgence=0 if _merge==2
replace interstate=0 if _merge==2
replace intrastate=0 if _merge==2
replace total_conflicts=0 if _merge==2 
replace battle_fat=0 if battle_fat==.




gen conflict_dummy=0
replace conflict_dummy=1 if total_intensity>=1

drop _merge
save counts_rights_demo_conflict, replace




/******************************* Adding the Geopolitical Variables to the Database********************************************/
import excel "C:\Users\yjl6\Dropbox (YLS)\Anti Semitism\Israel and the UN\Data\GPT_Counts\Geopol_1981_2023.xlsx", firstrow clear
rename Country country
rename Year year

replace country = trim(country)
drop if year<1981


/*Fixing up country names*/
replace country="Bosnia Herzegovenia" if country=="Bosnia and Herzegovina"
replace country="Cape Verde" if country=="Cabo Verde"
replace country="Congo, Republic of" if country=="Congo (Congo-Brazzaville)"
replace country="Congo, Democratic Republic of" if country=="Democratic Republic of the Congo"
replace country="Gambia, The" if country=="Gambia"
replace country="Kyrgyz Republic" if country=="Kyrgyzstan"
replace country="Micronesia, Federated States of" if country=="Micronesia"
replace country="Myanmar" if country=="Myanmar (formerly Burma)"
replace country="Korea, Democratic People's Republic of" if country=="North Korea"
replace country="Macedonia" if country=="North Macedonia"
replace country="Slovak Republic" if country=="Slovakia"
replace country="Korea, Republic of" if country=="South Korea"
replace country="East Timor" if country=="Timor-Leste"
replace country="United States of America" if country=="United States"



/*Fixing up occupation variable per RULAC*/
replace TerritoryOccupied=1 if country=="Armenia" /*See https://www.rulac.org/browse/countries/armenia*/
replace TerritoryOccupied=1 if country=="Ethiopia"&year>2002 /*See https://www.rulac.org/browse/countries/ethiopia*/


*Fixing up nuclear power outside NPT variable for South Africa, https://www.nti.org/countries/south-africa/
replace NuclearPowerOutsideNPT=1 if country=="South Africa"&year<1991 


sort country year

merge 1:1 country year using counts_rights_demo_conflict
drop if _merge==1

/*Fixing up countries missing variables from geopolitical data*/
replace OECDMember=0 if country=="Cote d'Ivoire"|country=="Moldova"|country=="Kosovo"
replace NATOMember=0 if country=="Cote d'Ivoire"|country=="Moldova"|country=="Kosovo"
replace NuclearPowerOutsideNPT=0 if country=="Cote d'Ivoire"|country=="Moldova"|country=="Kosovo"
replace  MajorNonNATOAlly=0 if country=="Cote d'Ivoire"|country=="Moldova"|country=="Kosovo"
replace TerritoryOccupied=0 if country=="Cote d'Ivoire"|country=="Moldova"|country=="Kosovo"
replace PermanentUNSCMember=0 if country=="Cote d'Ivoire"|country=="Moldova"|country=="Kosovo"


drop _merge


/***********************************************************Adding and fixing additional variables to the data******************************/

*Making the count of critical resolutions equal to zero when it is missing
replace count=0 if count==.

*Destring numerical variables. 
destring Populationtotal, replace




*Taking logs of count, adding a small constant to deal with the zero terms. 
gen log_count=log(count+.1)


*Taking logs of battle fatalities, adding a small constant to handle the zero terms.
gen log_fatal=log(battle_fat+1)

*creating geopolitical dummies, western affiliate reflects military alliance with US
gen western_affiliate=0
replace western_affiliate=1 if NATOMember==1|MajorNonNATOAlly==1|country=="Israel"

*Creating a category for Israel plus NATO
gen nato_plusI=0
replace nato_plusI=1 if NATOMember==1|country=="Israel"

*Creating a dummy variable for Israel, the group variable
gen Israel_dummy=0
replace Israel_dummy=1 if country=="Israel"

*Creating a dummy variable for South Africa
gen SA_dummy=0
replace SA_dummy=1 if country=="South Africa"

*Getting rid of years and countries for which there is no data
drop if year==2024



*Turning World Bank data in numerical vars
gen population=PopulationtotalSPPOPTOTL/1000
gen log_population=log(population)
gen rgdp_ppp=real(GDPPPPconstant2021internat)
gen gdp_pc=real(GDPpercapitacurrentUSNY)
gen log_gdppc=log(gdp_pc)
gen rgdp_pc=rgdp_ppp/population
gen egrowth_rate=real(GDPpercapitagrowthannual)
gen life_expec=real(Lifeexpectancyatbirthtotal)
gen hum_cap_index=real(HumancapitalindexHCIscale)

*Generating year and region dummies
*region is missing variables for 22 and 23
bysort country (year): gen region2021 = unsubreg if year == 2021
bysort country: egen last_region = mode(region2021)
replace unsubreg=last_region if year==2022|year==2023
replace unsubreg=999 if unsubreg==.
tabulate unsubreg, generate(region_dummy)
tabulate year, generate(year_dummy)


*Making indicator variables for whether a country is part of the west and lagging in human rights
* Step 1: Sort the data by group and income
sort western_affiliate human_rights_score

* Step 2: Calculate the rank and total number of observations within each group
bysort western_affiliate year: egen rank = rank(human_rights_score)
bysort western_affiliate year: egen total = count(human_rights_score)

* Step 3: Determine the quintile
gen quintile = ceil(5 * rank / total)
* Step 3: Determine the quintile
gen quartile = ceil(5 * rank / total)

*Equals one if in bottom quintile of US military allies that year for human rights score
gen western_laggard=0
replace western_laggard=1 if quartile==1&western_affiliate==1


* Step 1: Sort the data by group and income
sort OECDMember human_rights_score

* Step 2: Calculate the rank and total number of observations within each group
bysort OECDMember: egen rank2 = rank(human_rights_score)
bysort OECDMember: egen total2 = count(human_rights_score)

* Step 3: Determine the quintile
gen quintile2 = ceil(5 * rank2 / total2)

*Equals one if in bottom quintile of OECD that year for human rights score
gen oecd_laggard=0
replace oecd_laggard=1 if quintile2==1&OECDMember==1


* Step 1: Sort the data by group and income
sort nato_plusI year human_rights_score

* Step 2: Calculate the rank and total number of observations within each group
bysort nato_plusI year: egen rank3 = rank(human_rights_score)
bysort nato_plusI year: egen total3 = count(human_rights_score)

* Step 3: Determine the quintile
gen quintile3 = ceil(5 * rank3 / total2)

*Equals one if in bottom quintile of NATO plus Israel that year for human rights score
gen nato_I_laggard=0
replace nato_I_laggard=1 if quintile3==1&nato_plusI==1


/*Making Interaction terms*/

gen oecdXconflict=OECDMember*total_intensity
gen allyXconflict=western_affiliate*total_intensity

gen gdpXconflict=rgdp_pc*total_intensity



/*Making the data into panel data*/
encode country, gen(country_num)
tab country_num, gen(country_dum)
xtset country_num year


/*Imputing missing variables*/
/*
*Imputing missing population values with regression model
regress Populationtotal country_dum* year 
predict pop_predicted, xb
replace Populationtotal = hr_predicted if missing(Populationtotal)
*/

/*Imputing missing variables*/
*Population is missing 2021-23. Filling in with pop growth rate* 2020 value. 
bysort country_num (year): gen pop_2020 = Populationtotal if year == 2020
bysort country_num: egen last_pop = mode(pop_2020)
by country_num: gen pop_growth = D.Populationtotal/Populationtotal
bysort country_num: egen mean_pop_growth=mean(pop_growth)
replace Populationtotal=last_pop*(1+mean_pop_growth) if year==2021
replace Populationtotal=last_pop*(1+mean_pop_growth)^2 if year==2022
replace Populationtotal=last_pop*(1+mean_pop_growth)^3 if year==2023

/*Imputing missing variables*/
*some human rights scores are missing. Filling them in with most recent variables 
bysort country_num (year): gen physint2021 = physint_sum if year == 2021
bysort country_num: egen last_physint = mode(physint2021)
replace physint_sum=last_physint if year==2022|year==2023

bysort country_num (year): gen bbatro2021 = bbatrocity if year == 2021
bysort country_num: egen last_bbat = mode(bbatro2021)
replace bbatrocity=last_bbat if year==2022|year==2023

bysort country_num (year): gen repress2020 = repression_sum if year == 2020
bysort country_num: egen last_repress = mode(repress2020)
replace repression_sum=last_repress if year==2022|year==2023|year==2021
replace repression_sum=repression_sum[_n+1] if year==1995

bysort country_num (year): gen civpol2020 = civpol_sum if year == 2020
bysort country_num: egen last_civpol = mode(civpol2020)
replace civpol_sum=last_civpol if year==2022|year==2023|year==2021


*Imputing human_rights_score from some of its components when it is missing. 
*For 2020-2023
sum human_rights_score, detail
bysort country_num (year): gen hr2019 = human_rights_score if year == 2019
bysort country_num: egen last_hr = mode(hr2019)
replace human_rights_score=last_hr if year==2020|year==2021|year==2022|year==2023


*Creating a variable that indicates which component variables are missing
gen missing_phys=0
replace missing_phys=1 if physint_sum==.
gen missing_repress=0
replace missing_repress=1 if repression_sum==.
gen missing_civpol=0
replace missing_civpol=1 if civpol_sum==.



*Imputing with appropriate regression model based on components for other years, using model that has no missing components
regress human_rights_score physint_sum repression_sum civpol_sum if missing_phys==0&missing_repress==0&missing_civpol==0
predict hr_predicted, xb

*Imputing with appropriate regression model based on components for other years, using model that has no missing components
regress human_rights_score civpol_sum if missing_civpol==0
predict hr_predicted2, xb

*Imputing with appropriate regression model based on components for other years, using model that has no missing components
regress human_rights_score repression_sum if missing_repress==0
predict hr_predicted3, xb

*Imputing with appropriate regression model based on components for other years, using model that has no missing components
regress human_rights_score physint_sum if missing_phys==0
predict hr_predicted4, xb

* When all else fails, using a country dummy to impute human rights score
quietly regress human_rights_score country_dum* year
predict hr_predicted5, xb

replace human_rights_score = hr_predicted if missing(human_rights_score)
sum human_rights_score, detail

replace human_rights_score = hr_predicted2 if missing(human_rights_score)
sum human_rights_score, detail

replace human_rights_score = hr_predicted3 if missing(human_rights_score)
sum human_rights_score, detail

replace human_rights_score = hr_predicted4 if missing(human_rights_score)
sum human_rights_score, detail


replace human_rights_score = hr_predicted5 if missing(human_rights_score)


sum human_rights_score, detail


gen log_hum_rights=log(human_rights_score)

save final_data, replace



