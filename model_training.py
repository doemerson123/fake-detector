import os
import pickle
from utils.data_pipeline_utils import file_directory
from utils.modeling_utils import ModelTraining


def execute_experiments():
    # train all models using cartesian product of model_params and save artifacts
    mt = ModelTraining()

    permuted_model_params = mt.permute_model_parameters()

    results_df = mt.train_all_models("3 CNN 1 Dense", permuted_model_params)

    results_path = os.path.join(file_directory("artifact"), "results_df.pkl")

    # aggregate artifacts are saved to the root artifacts location

    with open(results_path, "wb") as output_file:
        pickle.dump(results_df, output_file)
    output_file.close()


if __name__ == "__main__":
    execute_experiments()
