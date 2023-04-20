""" Custom Exception for the project. Shows the filename and line number. """

from types import TracebackType
from typing import TypeAlias

ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]
OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]


class CustomException(Exception):
    def __init__(
        self,
        error_message: Exception,
        error_detail: OptExcInfo,
    ) -> None:
        super().__init__(error_message)
        self.error_message = CustomException.error_message_detail(
            error_message, error_detail)

    @staticmethod
    def error_message_detail(
        error: Exception,
        error_detail: OptExcInfo,
    ) -> str:
        """
        error: Exception object raise from module.
        error_detail: `sys.exc_info()`
        """
        _, _, exc_tb = error_detail
        if exc_tb is not None:
            # Extracting file name from exception traceback
            file_name = exc_tb.tb_frame.f_code.co_filename
            # Preparing error message
            message = f"Error: '{file_name}'[{exc_tb.tb_lineno}]: {error}"
        else:
            message = 'No error details available for the raised exception.'
        return message

    def __str__(self) -> str:
        return self.error_message

    def __repr__(self) -> str:
        """ Formatting object of AppException """
        return CustomException.__name__.__str__()
