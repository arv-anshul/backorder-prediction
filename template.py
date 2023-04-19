""" Template for data science project by `Anshul Raj Verma`. """

import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s"
)


@dataclass
class Template:
    project_name: str

    def get_template_files_list(self) -> list[Path]:
        root = Path('.')
        p_name = self.project_name

        list_of_files = [
            root / p_name / '__init__.py',
            root / p_name / 'logger.py',
            root / p_name / 'config.py',
            root / p_name / 'exception.py',
            root / p_name / 'utils.py',
            root / "requirements.txt",
            root / "setup.py",
            root / "main.py",
        ]
        return list_of_files

    def create_template(self, list_of_files: list[Path]):
        for file in list_of_files:
            parent, name = file.parent, file.name

            if not parent.exists():
                logging.info("Creating a new directory: '%s'", parent)
                parent.mkdir(exist_ok=True)

            if not file.exists():
                logging.info("Creating a new file %s in directory '%s'",
                             name, parent)
                file.touch()
            else:
                logging.info("Required file is already present '%s'", file)

    def _write_into_file(self, fp: Path, txt: str):
        """
        fp: File path to write the `txt`.
        txt: Text to write in the file.

        Raises `FileNotFoundError` if path not exists or path is not a file.
        """

        if not fp.exists():
            raise FileNotFoundError("Path doesn't exists.")
        if not fp.is_file():
            raise FileNotFoundError('Path is not a file.')

        with open(fp, 'w') as f:
            f.write(txt)

    def template_for_setup_py_file(
        self,
        project_name: str,
        version: str,
        author: str,
        author_email: str,
        description: str
    ) -> None:
        """ Write the content for `setup.py` file. """

        txt = f"""\
from setuptools import find_packages, setup

requirements_txt = 'requirements.txt'
REMOVE_PACKAGE = '-e .'

def get_requirements() -> list[str]:
    with open(requirements_txt) as req_file:
        req_list = req_file.readline()
    req_list = [req_name.replace('\\n', '') for req_name in req_list]

    if REMOVE_PACKAGE in req_list:
        req_list.remove(REMOVE_PACKAGE)
    return req_list

setup(name='{project_name}',
      version='{version}',
      description='{description}',
      author='{author}',
      author_email='{author_email}',
      packages=find_packages(),
      install_requires=get_requirements()
      )
"""

        fp = Path('./setup.py')
        logging.info('Writing in %s file.', fp)
        self._write_into_file(fp, txt)
        logging.info('Done Writing!')

    def template_for_logging(self):
        """ Write default template for `logger.py` file. """

        txt = """\
\""" Basic logging definition for this project. \"""

import logging
from datetime import date
from pathlib import Path

LOG_DIR_PATH = Path('logs')
LOG_DIR_PATH.mkdir(exist_ok=True)

LOG_FILE_PATH = LOG_DIR_PATH / (date.today().strftime('%d-%m-%Y') + '.log')

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(filename)s:[%(lineno)d] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
"""

        fp = Path('.', self.project_name, 'logger.py')
        logging.info('Writing in %s file.', fp)
        self._write_into_file(fp, txt)
        logging.info('Done Writing!')

    def template_for_exception(self):
        """ Write default template for `exception.py` file. """

        txt = """\
\""" Custom Exception for the project. Shows the filename and line number. \"""

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
        \"""
        error: Exception object raise from module.
        error_detail: `sys.exc_info()`
        \"""
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
        \""" Formatting object of AppException \"""
        return CustomException.__name__.__str__()
"""

        fp = Path('.', self.project_name, 'exceptions.py')
        logging.info('Writing in %s file.', fp)
        self._write_into_file(fp, txt)
        logging.info('Done Writing!')


class DataScienceTemplate(Template):

    def get_template_files_list(self) -> list[Path]:
        list_of_files = super().get_template_files_list()

        root = Path('.')
        p_name = self.project_name

        list_of_files.extend([
            root / p_name / '__init__.py',
            root / p_name / 'components' / '__init__.py',
            root / p_name / 'components' / 'data' / '__init__.py',
            root / p_name / 'components' / 'data' / 'ingestion.py',
            root / p_name / 'components' / 'data' / 'transformation.py',
            root / p_name / 'components' / 'data' / 'validation.py',
            root / p_name / 'components' / 'model' / '__init__.py',
            root / p_name / 'components' / 'model' / 'evaluation.py',
            root / p_name / 'components' / 'model' / 'pusher.py',
            root / p_name / 'components' / 'model' / 'trainer.py',
            root / p_name / 'entity' / '__init__.py',
            root / p_name / 'entity' / 'artifact_entity.py',
            root / p_name / 'entity' / 'config_entity.py',
            root / p_name / 'pipeline' / '__init__.py',
            root / p_name / 'pipeline' / 'prediction.py',
            root / p_name / 'pipeline' / 'training.py',
            root / p_name / 'predictor.py',
        ])
        return list_of_files


def ask_project_name() -> str:
    while True:
        p_name = input('Enter your project name: ')
        if p_name != '':
            check = input(f'Is your project name is {p_name!r} [y/n]: ')
            if check == 'y':
                break
    return p_name


def main():
    p_name = ask_project_name()
    logging.info(f'Creating project by name: {p_name}')

    ds_template = DataScienceTemplate(p_name)
    list_of_files = ds_template.get_template_files_list()
    ds_template.create_template(list_of_files)

    # Default templates for some files
    ds_template.template_for_setup_py_file(p_name, '0.0.1',
                                           'Anshul Raj Verma',
                                           'arv.anshul.1864@gmail.com',
                                           f'This is {p_name} project.')
    ds_template.template_for_logging()
    ds_template.template_for_exception()


# if __name__ == '__main__':
#     main()
