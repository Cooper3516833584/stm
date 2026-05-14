from loguru import logger

from .Application import FC_Application


class FC_Controller(FC_Application):
    """Local flight controller."""

    pass


FC_Like = FC_Controller

__all__ = [
    "FC_Controller",
    "FC_Like",
    "logger",
]
