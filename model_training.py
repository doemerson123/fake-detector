from utils.load_params_util import load_params
from utils.modeling_util import permute_model_parameters, train_all_models
from utils.data_pipeline_util import filepath
import pickle 

params = load_params('params.yaml')
max_epochs = params.model_training.model_params.max_epochs

permuted_model_params = permute_model_parameters()

# train all models using cartesian product of model_params and save artifacts
model_artifacts = train_all_models('3 CNN 1 Dense', 
                                    permuted_model_params, 
                                    max_epochs)

artifact_filename_list = ['results_df', 'model_list', 'model_name_list']

#set model artifact location
artifact_filepath, _ = filepath('artifact')

# artifacts are saved to model_artifacts location
for artifact, filename in zip(model_artifacts, artifact_filename_list):
    with open(f"{artifact_filepath + filename}.pkl", "wb") as output_file:
        pickle.dump(artifact, output_file)
    output_file.close


