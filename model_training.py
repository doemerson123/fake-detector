from utils.load_params import load_params
from utils.modeling import permute_model_parameters, train_all_models
import pickle 

params = load_params('params.yaml')
max_epochs = params.model_training.model_params.max_epochs

permuted_model_params = permute_model_parameters()

# train all models using all permuations of model_params and save aggregated artifacts
model_artifacts = train_all_models('3 CNN 1 Dense', permuted_model_params, max_epochs)
artifact_filename_list = ['results_df', 'model_list', 'model_name_list']

#set model artifact location
training_locally = params.data_pipeline.training_locally
if training_locally: 
    filepath = params.data_pipeline.pipeline_local_filepath
else: 
    filepath = params.data_pipeline.pipeline_cloud_filepath

# artifacts are saved to model_artifacts location
for artifact, filename in zip(model_artifacts, artifact_filename_list):
    with open(f"{filepath + filename}.pkl", "wb") as output_file:
        pickle.dump(artifact, output_file)
    output_file.close


