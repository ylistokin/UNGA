# UNGA
python files used to create data for analysis for UN project


Input 1: The initial full text of each UNGA resolution from 1994 to the presentdata from Voelsen et al. 

Inpute 2: The initial full text of each UNGA resolution from 1981 to 1993 Mesquita.


The .ipynb file "/content/gdrive/MyDrive/UNGA/GPT on full Voelsen" runs the text of each resolution through ChatGPT 4o mini and determines if each resolution criticizes a particular country. The resulting determinations are saved in df = pd.read_csv("/content/gdrive/MyDrive/UNGA/full_text_data2.csv")

The .ipynb file "/content/gdrive/MyDrive/UNGA/GPT on Mesquita" runs the text of each resolution from 1981-1993 through ChatGPT 4o mini and determines if each resolution criticizes a particular country. The resulting determinations are saved in df = pd.read_csv("/content/gdrive/MyDrive/UNGA/full_text_data2.csv")

I calculate the number of annual critical resolutions for each country from 1981-2023 in /content/gdrive/MyDrive/UNGA/Annual Country Resolutions MV.ipynb") (This file combines the Mesquita and Voelsen data together.) I save the resulting data as /content/gdrive/MyDrive/UNGA/GPT_resolution_counts.csv This pythong file also separates out "unique". I also save a separate file that drops resolutions with multiple sections counted as separate resolutions in the file as /content/gdrive/MyDrive/UNGA/GPT_unique_resolution_counts.csv

I calculate the number of annual critical resolutions for each country from 1981-1993 in /content/gdrive/MyDrive/UNGA/v2Mes Annual Country Resolutions.ipynb").
