import os


# Options of CNN model
class ModelOptions:
    _COMMON_FILE_NAME: str = 'models_summary'
    _DATA_DIRECTORY: str = 'data/'

    def __init__(self, name: str, input_shape: tuple[int, int, int], drop_conv: float = 0.0, drop_fc: float = 0.0) -> None:
        """ Initialize CNN parameters."""
        assert 0 < drop_conv <= 1, 'Drop chance for convolution layer must be in range (0,1]'
        assert 0 < drop_fc <= 1, 'Drop chance for full connected layer must be in range (0,1]'
        assert all([i > 0 for i in input_shape]), 'Input shape values mst be > 0'
        self.name: str = name
        self.drop_conv: float = drop_conv
        self.drop_fc: float = drop_fc
        self.input_shape: (int, int, int) = input_shape
        self.history: {} = None

    def get_model_png_file_name(self) -> str:
        """ Get png file name to save history of model."""
        return self.__class__._DATA_DIRECTORY + self.name + '.png'

    @classmethod
    def get_common_xlsx_file_name(cls) -> str:
        """ Get xlsx file name to save history of model."""
        return cls._DATA_DIRECTORY + cls._COMMON_FILE_NAME + '.xlsx'

    @classmethod
    def get_common_png_file_name(cls) -> str:
        """ Get png file name to save history of all models."""
        return cls._DATA_DIRECTORY + cls._COMMON_FILE_NAME + '.png'

    @classmethod
    def create_data_directory(cls) -> None:
        """ Create directory of data."""
        directory = os.curdir + '/' + cls._DATA_DIRECTORY
        if not os.path.isdir(directory):
            os.mkdir(directory)
