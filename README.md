This toolkit enables the extraction, clipping, and processing of VIIRS night-time lights (NTL) data for a selected year and country. It generates GeoTIFFs clipped to administrative boundaries and produces CSV files containing zonal statistics based on user-defined administrative zones. 

🔐 Authentication and Data Access 

To use this script, you must register and obtain API credentials from the following data providers: 

1. NOAA Earth Observation Group (EOG) 

- Required to download the VIIRS annual ephemeral mask data (e.g., lit_mask.dat.tif.gz). 

- Register here: https://eogauth-new.mines.edu/realms/eog/protocol/openid-connect/auth 

- After registration, log in and use your username and password. Email EOG to obtain a client_id and secret. 

2. NASA Black Marble 

- You must register at NASA Earthdata: 

Sign up here: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#tokens 

- Access token is required when downloading via wget or requests. 

 

📁 File Structure 

- monthly_ntl_clean_int_v3_base.py – Python script handling extraction, filtering of poor pixels, masking, linear imputation, and zonal statistics logic 

- ntl_monthly_bmi_redacted.ipynb – Notebook interface (recommended: run on Google Colab) 

- Shapefile/ – Folder containing your admin boundary shapefile 


🛠️ Prerequisites 

- Prepare the following and upload them into your Colab session: 

- A shapefile folder named Shapefile 

- CSV tileset of the country of interest 


🚀 How to Use 

- Open ntl_monthly_bmi.ipynb in Google Colab 

- Upload the prerequisite files (see above) 

- Run the cells in order 
