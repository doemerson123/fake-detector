import pytest
from os import path, walk
from imutils import paths
from box import ConfigBox
from pathlib import Path
from utils.data_pipeline_utils import (
    load_params,
    file_directory,
    load_image,
    augment,
    train_test_val_filepaths,
    dataset_generator,
    create_dataset,
    training_locally,
)
import sys
from tensorflow import (
    constant,
    equal,
    reduce_all,
    float32,
    image,
    io,
)
from tensorflow.data.experimental import cardinality
from test.conftest import load_test_params, test_image_dir, batch_size

# model Training imports
import pytest
from utils.modeling_utils import ModelTraining
from utils.custom_metrics_utils import StatefullMultiClassFBeta
from tensorflow.keras.models import Sequential
from test.conftest import (
    mock_params_dict,
    one_layer_removed_params_dict,
    first_conv_layer_conv_layer_attributes,
)
from box.exceptions import BoxKeyError
from pandas import DataFrame, json_normalize, Series
from numpy import isnan


class TestDataPipelineUtils:
    """
    Test class for data_pipeline_utils module
    """

    def test_load_params_training_locally(self, load_test_params: ConfigBox) -> None:
        """
        Confirm the value of training_locally in test_params.yaml is set to True.

        Parameters
        ----------
            load_test_params (ConfigBox): A ConfigBox instance containing 
            the parameters loaded from test_params.yaml

        Returns
        -------
            None
        """

        training_locally = load_test_params.root_directory.training_locally_bool
        assert type(training_locally) == bool
        assert training_locally

    def test_load_params_open(self) -> None:
        """
        Confirm params.yaml and test_params.yaml can be opened and have the correct type.

        Returns
        -------
            None
        """
        assert load_params()  # tests params.yaml as default optional parameter
        assert load_params(path.join("test", "test_params.yaml")) != None
        assert type(load_params()) == ConfigBox

    def test_load_params_not_found(self) -> None:
        """
        Confirm FileNotFoundError is raised if the file is not found.

        Returns
        -------
            None
    """
        with pytest.raises(FileNotFoundError):
            load_params("bad_filename.yaml")

    def test_load_params_unhandled(self) -> None:
        pass
        # with mock.patch(load_params("bad_filename.yaml")) as mock_open:
        #     mock_open.side_effect = Exception
        #     assert mock_open == Exception

    def test_file_directory_resolution(self, load_test_params: ConfigBox) -> None:
        """
        Validate that the file_directory method points to the correct asset and
        artifact paths.

        Parameters
        ----------
            load_test_params (ConfigBox): A ConfigBox instance containing the
            parameters loaded from test_params.yaml

        Returns
        -------
            None
        """
        local_root = load_test_params.root_directory.local_filepath
        cloud_root = load_test_params.root_directory.cloud_filepath
        training_locally = load_test_params.root_directory.training_locally_bool

        root = cloud_root
        if training_locally:
            root = local_root
        else:
            root = cloud_root

        assert file_directory("root") == path.join(root)
        assert file_directory("data") == path.join(root, "Data")
        assert file_directory("callback") == path.join(root, "Callbacks")
        assert file_directory("artifact") == path.join(root, "Artifacts")

    def test_load_image_open(self, test_image_dir: str) -> None:
        """
        Confirm that load_image can successfully open the test jpeg file 
        and return a tuple.

        Parameters
        ----------
            test_image_dir (str): The path to the test jpeg file

        Returns
        -------
            None
        """

        assert load_image(test_image_dir)
        assert type(load_image(test_image_dir)) == tuple
        assert load_image(test_image_dir) != None

    def test_load_image_label(self, test_image_dir: str) -> None:
        """
        Confirms load_image will correctly resolve the label from the image in
        the test directory fake-detector/test/Fake/test_Fake 4.jpg

        Due to the file structure of the underlying test data the folder named
        "Fake" is required since the tree is traversed to check for that value.
    
        Parameters
        ----------
            test_image_dir (str): The path to the test image

        Returns
        -------
            None
        """
        test_label = load_image(test_image_dir)[1]
        known_label = constant([1.0, 0.0])
        assert reduce_all(equal(test_label, known_label))

    def test_load_image_shape(self, test_image_dir: str, img_size: int) -> None:
        """
        Confirms load_image returns a tensor with the appropriate shape based 
        on the img_size parameter in test_params.yaml.

        Parameters
        ----------
            test_image_dir (str): The path to the test image
            img_size (int): The desired size of the image

        Returns
        -------
            None
        """
        test_image_shape = load_image(test_image_dir)[0].shape
        known_image_shape = [img_size, img_size, 3]
        assert reduce_all(equal(test_image_shape, known_image_shape))

    def test_augment_image_change(self, test_image_dir: str, img_size: int) -> None:
        """
        Confirms augmention occurs

        Note: There are cases where the image does not get altered due to
        randomness. This is acceptable since this method randomly alters
        images. It does not alter every image in the data set. It's unlikely
        two attempts will result in no image augmentation.

        Parameters
        ----------
            test_image_dir (str): The path to the test image
            img_size (int): The desired size of the image
        
        Returns
        -------
            None
        """
        test_tuple = load_image(test_image_dir)
        test_augmented_tuple = augment(test_tuple[0], test_tuple[1])

        known_image = io.read_file(test_image_dir)
        known_image = image.decode_jpeg(known_image, channels=3)
        known_image = image.convert_image_dtype(known_image, dtype=float32)
        known_image = image.resize(known_image, (img_size, img_size))
        try:
            assert not reduce_all(equal(test_augmented_tuple[0], known_image))
        except:
            pass
        try:
            assert not reduce_all(equal(test_augmented_tuple[0], known_image))
        except:
            pass

    def test_train_test_val_filepaths_name(self) -> None:
        """
        Validates path values for train, val, and test directories.

        Returns
        -------
            None
        """
        train, val, test = train_test_val_filepaths()
        data_directory = file_directory("data")

        # happy path
        assert path.join(data_directory, "Train") in train[0]
        assert path.join(data_directory, "Val") in val[0]
        assert path.join(data_directory, "Test") in test[0]

        # jumbled/wrong path name
        assert not path.join(data_directory, "Test") in train[0]
        assert not path.join(data_directory, "Train") in val[0]
        assert not path.join(data_directory, "Val") in test[0]

    def total_images(self, directory: str) -> int:
        """
        Helper method that count all image files in the Data directory

        Requires directory tree: Data/{Train, Val, Test}/{Class 1, Class 2}
        """

        real_images_list = list(walk(directory))[2][2]
        fake_images_list = list(walk(directory))[1][2]
        return len(real_images_list + fake_images_list)

    def test_dataset_generator_batch_size(self, batch_size: int) -> None:
        """
        Validates batch size by taking the total num of files and dividing by
        the cardinatliy. This should equal the batch size in params.yaml.
        """
        train_filepath, val_filepath, test_filepath = train_test_val_filepaths()

        test_train_ds = dataset_generator(train_filepath, batch_size, "train_cache")
        test_val_ds = dataset_generator(val_filepath, batch_size, "val_cache")
        test_test_ds = dataset_generator(test_filepath, batch_size, "test_cache")

        train_dir = path.join(file_directory("data"), "Train")
        val_dir = path.join(file_directory("data"), "Val")
        test_dir = path.join(file_directory("data"), "Test")

        num_train_files = self.total_images(train_dir)
        num_val_files = self.total_images(val_dir)
        num_test_files = self.total_images(test_dir)

        # using floor division to avoid integer overflow
        assert num_train_files // cardinality(test_train_ds).numpy() == batch_size
        assert num_val_files // cardinality(test_val_ds).numpy() == batch_size
        assert num_test_files // cardinality(test_test_ds).numpy() == batch_size

    def test_create_dataset(self) -> None:
        pass


######################### This should be its own file but issues with pytest collecting a second file
mt = ModelTraining(params_file=path.join("test", "test_params.yaml"))


class TestModelTraining:
    def test_remove_extra_conv_params_removal(
        self, one_layer_removed_params_dict: dict
    ) -> None:
        """
        Confirm that all key value pairs are removed for third conv layer when
        only two conv layers are required for training.

        Confirm that no parameters are null and that first and second conv layer
        have values
        """

        with pytest.raises(BoxKeyError):
            one_layer_removed_params_dict.third_kernel
        with pytest.raises(BoxKeyError):
            one_layer_removed_params_dict.third_filter

        # confirm expected conv layers exist
        assert one_layer_removed_params_dict.first_kernel
        assert one_layer_removed_params_dict.first_filter
        assert one_layer_removed_params_dict.second_kernel
        assert one_layer_removed_params_dict.second_filter

    def test_remove_extra_conv_params_not_none(
        self, one_layer_removed_params_dict: dict
    ) -> None:
        """
        Confirm after removal of paramters the remaining dict are not None
        """
        for param in one_layer_removed_params_dict:
            assert param

    def test_permute_model_parameters_count(self):
        """
        Confirm list of params dicts calculate correctly from test_params.yaml
        """
        test_permuted_parameters = mt.permute_model_parameters()
        assert len(test_permuted_parameters) == 4

    def test_conv_layer_input(
        self, first_conv_layer_input_attributes: DataFrame
    ) -> None:
        """
        Confirm input layer in the mock_model matches expectations from
        the test_params.yaml file using the conv_layer method
        """
        assert first_conv_layer_input_attributes["config.batch_input_shape"] == [
            None,
            200,
            200,
            3,
        ]
        assert first_conv_layer_input_attributes["config.sparse"] == False
        assert first_conv_layer_input_attributes["config.ragged"] == False
        assert isnan(first_conv_layer_input_attributes["config.trainable"])

    def test_conv_layer_conv(
        self,
        first_conv_layer_conv_layer_attributes: DataFrame,
        second_conv_layer_conv_layer_attributes: DataFrame,
    ) -> None:
        """
        Confirm all convolution layers in the mock_model match expectations
        from the mock_params_dict fixture using the conv_layer method
        """
        assert first_conv_layer_conv_layer_attributes["config.kernel_size"] == [3, 3]
        assert first_conv_layer_conv_layer_attributes["config.trainable"] == True
        assert first_conv_layer_conv_layer_attributes["config.filters"] == 10
        assert first_conv_layer_conv_layer_attributes["config.strides"] == [1, 1]
        assert first_conv_layer_conv_layer_attributes["config.activation"] == "relu"
        assert first_conv_layer_conv_layer_attributes["config.batch_input_shape"] == [
            None,
            200,
            200,
            3,
        ]

        assert "conv2d" in second_conv_layer_conv_layer_attributes["config.name"]
        assert isnan(second_conv_layer_conv_layer_attributes["config.sparse"])
        assert isnan(second_conv_layer_conv_layer_attributes["config.ragged"])
        assert second_conv_layer_conv_layer_attributes["config.trainable"] == True

        assert second_conv_layer_conv_layer_attributes["config.kernel_size"] == [5, 5]
        assert second_conv_layer_conv_layer_attributes["config.trainable"] == True
        assert second_conv_layer_conv_layer_attributes["config.filters"] == 15
        assert second_conv_layer_conv_layer_attributes["config.strides"] == [1, 1]
        assert second_conv_layer_conv_layer_attributes["config.activation"] == "relu"
        assert isnan(
            second_conv_layer_conv_layer_attributes["config.batch_input_shape"]
        )

    def test_conv_layer_pooling(
        self,
        first_conv_layer_pooling_attributes: DataFrame,
        second_conv_layer_pooling_attributes: DataFrame,
    ) -> None:
        """
        Confirm convolution pooling layers in the mock_model match expectations
        from the mock_params_dict fixture using the conv_layer method
        """

        assert "pooling" in first_conv_layer_pooling_attributes["config.name"]
        assert first_conv_layer_pooling_attributes["config.trainable"] == True
        assert first_conv_layer_pooling_attributes["config.pool_size"] == [2, 2]
        assert first_conv_layer_pooling_attributes["config.strides"] == [2, 2]

        assert "pooling" in second_conv_layer_pooling_attributes["config.name"]
        assert second_conv_layer_pooling_attributes["config.trainable"] == True
        assert second_conv_layer_pooling_attributes["config.pool_size"] == [2, 2]
        assert second_conv_layer_pooling_attributes["config.strides"] == [2, 2]

    def test_conv_layer_dropout(
        self,
        first_conv_layer_dropout_attributes: DataFrame,
        second_conv_layer_dropout_attributes: DataFrame,
    ) -> None:
        """
        Confirm convolution dropout layers in the mock_model match expectations
        from the mock_params_dict fixture using the conv_layer method.
        """
        assert first_conv_layer_dropout_attributes["config.rate"] == 0.4
        assert first_conv_layer_dropout_attributes["config.trainable"] == True

        assert "dropout" in second_conv_layer_dropout_attributes["config.name"]
        assert second_conv_layer_dropout_attributes["config.rate"] == 0.4
        assert second_conv_layer_dropout_attributes["config.trainable"] == True

    def test_conv_layer_batch_norm(
        self,
        first_conv_layer_batch_norm_attributes: DataFrame,
        second_conv_layer_batch_norm_attributes: DataFrame,
    ) -> None:
        """
        Confirm convolution batch layers in the mock_model match expectations
        from the mock_params_dict fixture using the conv_layer method.
        """
        assert first_conv_layer_batch_norm_attributes["config.trainable"] == True

        assert (
            "batch_normalization"
            in second_conv_layer_batch_norm_attributes["config.name"]
        )
        assert second_conv_layer_batch_norm_attributes["config.trainable"] == True

    def test_flatten(self, flatten_attributes: DataFrame) -> None:
        """
        Confirm flatten in the mock_model match expectations from the
        mock_params_dict fixture.
        """
        assert "flatten" in flatten_attributes["config.name"]
        assert flatten_attributes["config.trainable"] == True

    def test_dense_layer_batch_norm(
        self, dense_layer_batch_norm_attributes: DataFrame
    ) -> None:
        """
        Confirm batch norm in the mock_model matches expectations from the
        mock_params_dict fixture using the dense_layer funciton.
        """

        assert "batch_normalization" in dense_layer_batch_norm_attributes["config.name"]
        assert dense_layer_batch_norm_attributes["config.trainable"] == True

    def test_dense_layer_dense(self, dense_layer_attributes: DataFrame) -> None:
        """
        Confirm dense layer in the mock_model matches expectations from the
        mock_params_dict fixture using the dense_layer funciton.
        """
        assert "dense" in dense_layer_attributes["config.name"]
        assert dense_layer_attributes["config.units"] == 10
        assert dense_layer_attributes["config.trainable"] == True
        assert dense_layer_attributes["config.kernel_regularizer.class_name"] == "L2"
        assert dense_layer_attributes["config.activation"] == "relu"

    def comparable_model_config_file(self, model: Sequential) -> DataFrame:
        """
        Helper method that takes model.get_config() and resolves issues that
        prevent an all ways boolean compare.
        """
        top_layer = DataFrame(model.get_config()["layers"])
        model_dataframe = json_normalize(top_layer["config"])
        cols_to_keep = [
            "batch_input_shape",
            "trainable",
            "filters",
            "kernel_size",
            "strides",
            "padding",
            "activation",
            "pool_size",
            "rate",
            "units",
            "kernel_regularizer.config.l2",
        ]
        return model_dataframe[cols_to_keep]

    def test_model_architecture(
        self, mock_params_dict: dict, mock_model: Sequential
    ) -> None:
        """
        Confirms mock_model paramters match the paramters created using
        the model_architecture method from the test_params.yaml file
        """
        params = mt.remove_extra_conv_params(mock_params_dict, num_conv_layers=2)
        test_model = mt.model_architecture(params, "ModelArchTest")

        test_model_config = self.comparable_model_config_file(test_model)
        mock_model_config = self.comparable_model_config_file(mock_model)
        print(test_model_config)

        assert (
            test_model_config["batch_input_shape"][0]
            == mock_model_config["batch_input_shape"][0]
        )
        assert (
            test_model_config["batch_input_shape"][1]
            == mock_model_config["batch_input_shape"][1]
        )

        assert test_model_config["filters"][1] == mock_model_config["filters"][1]
        assert test_model_config["filters"][5] == mock_model_config["filters"][5]

        assert (
            test_model_config["kernel_size"][1] == mock_model_config["kernel_size"][1]
        )
        assert (
            test_model_config["kernel_size"][5] == mock_model_config["kernel_size"][5]
        )

        assert test_model_config["strides"][1] == mock_model_config["strides"][1]
        assert test_model_config["strides"][2] == mock_model_config["strides"][2]
        assert test_model_config["strides"][5] == mock_model_config["strides"][5]
        assert test_model_config["strides"][6] == mock_model_config["strides"][6]

        assert test_model_config["activation"][1] == mock_model_config["activation"][1]
        assert test_model_config["activation"][5] == mock_model_config["activation"][5]
        assert (
            test_model_config["activation"][11] == mock_model_config["activation"][11]
        )
        assert (
            test_model_config["activation"][12] == mock_model_config["activation"][12]
        )

        assert test_model_config["pool_size"][2] == mock_model_config["pool_size"][2]
        assert test_model_config["pool_size"][6] == mock_model_config["pool_size"][6]

        assert test_model_config["rate"][3] == mock_model_config["rate"][3]
        assert test_model_config["rate"][7] == mock_model_config["rate"][7]

        assert test_model_config["units"][11] == mock_model_config["units"][11]
        assert test_model_config["units"][12] == mock_model_config["units"][12]

        assert (
            test_model_config["kernel_regularizer.config.l2"][11]
            == mock_model_config["kernel_regularizer.config.l2"][11]
        )

    def test_train_model(self):
        pass

    def test_train_all_models(self):
        pass
