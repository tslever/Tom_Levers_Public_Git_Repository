import pandas as pd
curated_data_for_exoplanets = pd.read_csv("6--Curated_Data_For_Exoplanets.csv")
mask_for_density = ~curated_data_for_exoplanets['density_in_grams_per_cubic_centimeter'].isnull()
mask_for_semimajor_axis = ~curated_data_for_exoplanets['orbital_semimajor_axis_in_AU'].isnull()
curated_data_for_exoplanets_with_winnowed_density_and_semimajor_axis = curated_data_for_exoplanets[mask_for_density & mask_for_semimajor_axis]
curated_data_for_exoplanets_with_winnowed_density_and_semimajor_axis.to_csv('7--Curated_Data_With_Density_And_Semimajor_Axis.csv')
