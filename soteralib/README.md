# Data Loading
## level2
```Python
from sotodlib.io.load_smurf import Observations
import soteralib as tera
SMURF, session, obs_all = tera.load_data.get_obs_all_level2(telescope)
obs_all = obs_all.filter(Observations.start > dt.datetime(2024,2,8),
                         Observations.stream_id == 'ufm_mv5', 
                        Observations.calibration == False)
aman = tera.load_data.load_data_level2(obs_id, telescope='satp3')
```

## level3
```Python
import soteralib as tera
ctx_file = 'context.yaml'
query_line = 'start_time > 1704200000 and type == "obs"'
query_tags = ['jupiter=1', 'rising=1']
tera.load_data.get_obsdict_level3(ctx_file, query_line=query_line, query_tags=query_tags)
```

