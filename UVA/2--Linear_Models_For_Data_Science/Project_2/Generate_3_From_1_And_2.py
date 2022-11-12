import pandas as pd

data_for_exoplanets_from_planetary_systems_data_frame = pd.read_csv("Data_For_Exoplanets_From_Planetary_Systems_Data_Frame.csv")
#print(data_for_exoplanets_from_planetary_systems_data_frame['pl_name'])

data_for_terrestrial_exoplanets_from_exoplanets_catalog = pd.read_csv("Data_For_Terrestrial_Exoplanets_From_Exoplanets_Catalog.csv")
#print(data_for_terrestrial_exoplanets_from_exoplanets_catalog['display_name'])

data_for_terrestrial_exoplanets = data_for_exoplanets_from_planetary_systems_data_frame.merge(data_for_terrestrial_exoplanets_from_exoplanets_catalog, how = 'inner', left_on = "pl_name", right_on = "display_name")
data_for_terrestrial_exoplanets.to_csv("Data_For_Terrestrial_Exoplanets.csv")

data_for_nonterrestrial_exoplanets = data_for_exoplanets_from_planetary_systems_data_frame.loc[~data_for_exoplanets_from_planetary_systems_data_frame['pl_name'].isin(list(data_for_terrestrial_exoplanets_from_exoplanets_catalog['display_name']))]
data_for_nonterrestrial_exoplanets.to_csv('Data_For_Non-Terrestrial_Exoplanets.csv')

data_for_terrestrial_exoplanets = data_for_exoplanets_from_planetary_systems_data_frame.merge(data_for_terrestrial_exoplanets_from_exoplanets_catalog, how = 'outer', left_on = "pl_name", right_on = "display_name")
data_for_terrestrial_exoplanets.to_csv('Data_For_Exoplanets.csv')