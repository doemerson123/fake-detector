from load_params import load_params

yamll = load_params("params.yaml")
print(yamll.data_collection.target_website)

