# VS Code Demo File

#%% 
import pandas as pd
import numpy as np


#%% Create a simple DataFrame
a = np.random.random(10)
b = np.random.random(10)
df = pd.DataFrame(dict(a=a, b=b))
df.index.name = 'obs_id'

# %%
