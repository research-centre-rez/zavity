from config.config import OUTPUT_FOLDER
import os
import numpy as np
import pandas as pd
import logging


def dump_path(video_name, object_name, extension='npy'):
    """
    Generates a path for saving or loading a specific object related to the video.

    Args:
        video_name (str): Name of the input (used as prefix)
        object_name (str): Name of the object to save/load.
        extension (str): File extension for the object. Defaults to 'npy'.

    Returns:
        str: Path to the file.
    """
    return os.path.join(OUTPUT_FOLDER, os.path.splitext(video_name)[0] + f'-{object_name}.{extension}')


def dump(prefix: str, suffix: str, obj):
    """
    Saves an object to a file using numpy save function.

    Args:
        prefix (str): Filename prefix
        suffix (str): Filename suffix.
        obj: The object to save.
    """
    np.save(dump_path(prefix, suffix), obj)


def dump_csv(prefix: str, suffix: str, dataFrame: pd.DataFrame) -> None:
    logging.debug(f"Writing file: {dump_path(prefix, suffix, extension="csv")}")
    dataFrame.to_csv(dump_path(prefix, suffix, extension="csv"), index=False)


def load_csv(prefix: str, suffix: str) -> np.ndarray:
    array2D = pd.read_csv(dump_path(prefix, suffix, extension="csv")).to_numpy()
    return array2D.reshape(-1) if array2D.shape[-1] == 1 else array2D

def file_exists(prefix: str, suffix: str, extension='npy') -> bool:
    return os.path.isfile(dump_path(prefix, suffix, extension=extension))
