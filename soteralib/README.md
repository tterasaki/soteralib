# Data Loading
## level2
```Python
import soteralib
SMURF, session, obs_all = soteralib.load_data.get_obs_all_level2(telescope)
obs_all = obs_all.filter(Observations.start > dt.datetime(2024,2,8),
                         Observations.stream_id == 'ufm_mv5', 
                        Observations.calibration == False)
aman = soteralib.load_data.load_data_level2(obs_id, telescope='satp3')
```

## level3
```Python
ctx_file = 'context.yaml'
query_line = 'start_time > 1704200000 and type == "obs"'
query_tags = ['jupiter=1', 'rising=1']
soteralib.load_data.get_obsdict(ctx_file, query_line=None, query_tags=None)
```

