import os
import pickle
from utils.data_pipeline_utils import file_directory
from utils.modeling_util import ModelTraining


# train all models using cartesian product of model_params and save artifacts
mt = ModelTraining()
permuted_model_params = mt.permute_model_parameters()

model_artifacts = mt.train_all_models("3 CNN 1 Dense", permuted_model_params)

artifact_path_list = []
artifact_name_list = ["results_df", "model_list", "model_name_list"]
for file in artifact_name_list:
    artifact_path_list.append(os.path.join(file_directory("artifact"), file + ".pkl"))

# aggregate artifacts are saved to the root artifacts location
for artifact, path in zip(model_artifacts, artifact_path_list):
    with open(path, "wb") as output_file:
        pickle.dump(artifact, output_file)
    output_file.close()
