# Import the HfApi class and ModelFilter from the huggingface_hub library
from huggingface_hub import HfApi, ModelFilter 
# Instantiate the HfApi class
api = HfApi()
# Retrieve the first 5 models
models = api.list_models(
    filter=ModelFilter(
        task="text-classification"),
        sort="downloads",
        direction=-1,
        limit=5  
        
    )
)
modelList = list(models)
for mode in modelList:
  print(mode.modelId)
