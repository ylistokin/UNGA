/*This file takes the data on the UN and control variables and gives summary statistics*/

use final_data, clear


*Getting means for data discussed in the text of Section IV of the paper
sum count if Israel_dummy==1, detail

sum count if SA_dummy==1&year<1990, detail



*Making Table 1
collect clear
sort Israel_dummy
collect: by Israel_dummy: mean count total_intensity battle_fat TerritoryOccupied human_rights_score NuclearPowerOutsideNPT OECDMember western_affiliate population gdp_pc


collect label levels colname count "Critical UNGA Resolutions" total_intensity "War Intensity" battle_fat "War Fatalities" TerritoryOccupied "Occupying Country Indicator" human_rights_score "Human Rights Score" NuclearPowerOutsideNPT "Nuclear Weapons Outside of Treaty" OECDMember "Member of OECD" western_affiliate "US Military Ally" population "Population (1000s)" gdp_pc "GDP per capita (current $)"  , modify
collect label levels Israel_dummy 0 "Other Countries" 1 "Israel", modify
collect label levels result _r_b "Mean" sd "Standard Dev." _N "Total Nonmissing Observations", modify
collect style cell result, nformat(%5.1f)
collect style cell colname[TerritoryOccupied]#result[_r_b sd], nformat(%5.2f)
collect style cell colname[NuclearPowerOutsideNPT]#result[_r_b sd], nformat(%5.2f)
collect style cell colname#result[_N], nformat(%5.0f)
collect title Table 1: Annual Summary Statistics for Israel and 193 Other Countries (1981-2023)
collect notes Annual mean and standard devation of key variables (by country) for the years 1981-2023


collect layout (colname[count total_intensity battle_fat TerritoryOccupied human_rights_score NuclearPowerOutsideNPT OECDMember western_affiliate population]#result[_r_b sd] colname[gdp_pc]#result[_r_b sd _N]) (Israel_dummy[1 0])

collect export table1.docx, replace