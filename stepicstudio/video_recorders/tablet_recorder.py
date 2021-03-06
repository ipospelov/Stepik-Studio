import logging
import time

from django.conf import settings
from singleton_decorator import singleton

from stepicstudio.operations_statuses.operation_result import InternalOperationResult
from stepicstudio.operations_statuses.statuses import ExecutionStatus
from stepicstudio.ssh_connections.tablet_client import TabletClient

ATTEMPTS_TO_STOP = 30
ATTEMPTS_PAUSE = 0.1  # seconds


@singleton
class TabletScreenRecorder(object):
    def __init__(self):
        self.__command = settings.FFMPEG_TABLET_CMD
        self.__logger = logging.getLogger('stepic_studio.video_recorders.tablet_recorder')
        self.last_processed_path = None
        self.last_processed_file = None

        try:
            self.__tablet_client = TabletClient(connect_timeout=10)
        except Exception as e:
            self.__logger.error('Can\'t initialize tablet client: %s', e)
            raise e

    def start_recording(self, path: str, filename: str) -> InternalOperationResult:
        if self.is_active():
            self.__logger.error('Can\'t start FFMPEG for file %s: screen is acctually recording ',
                                path + '/' + filename)
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR, 'Tablet is actually recording')

        try:
            self.__tablet_client.check_and_create_folder(path)
        except Exception as e:
            self.__logger.error('Error while creating remmote dir: %s', e)
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR)

        command = settings.FFMPEG_TABLET_CMD + path + '/' + filename + ' 2< /dev/null &'

        try:
            self.__tablet_client.execute_remote(command)
        except Exception as e:
            self.__logger.error('Screen recording start failed: %s; FFMPEG command: %s', e, command)
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR)

        self.last_processed_file = filename
        self.last_processed_path = path
        self.__logger.info('Successfully start screen recording (FFMPEG command: %s', command)
        return InternalOperationResult(ExecutionStatus.SUCCESS)

    def stop_recording(self) -> InternalOperationResult:
        if not self.is_active():
            self.__logger.warning('Tablet screencast isn\'t active: can\'t stop non existing FFMPEG process')
            return InternalOperationResult(ExecutionStatus.SUCCESS)

        command = 'pkill -f ffmpeg'

        for _ in range(0, ATTEMPTS_TO_STOP):
            try:
                #  read_output=True is using for synchronized execution
                self.__tablet_client.execute_remote(command, allowable_code=1, read_output=True)
            except Exception as e:
                self.__logger.error('Problems while stop screen recording: %s', e)
                return InternalOperationResult(ExecutionStatus.FATAL_ERROR)

            if not self.is_active():
                return InternalOperationResult(ExecutionStatus.SUCCESS)
            else:
                time.sleep(ATTEMPTS_PAUSE)

        self.__logger.error('Can\'t stop screen recording process for %s seconds.', ATTEMPTS_PAUSE * ATTEMPTS_TO_STOP)
        return InternalOperationResult(ExecutionStatus.FIXABLE_ERROR)

    def is_active(self) -> bool:
        try:
            output = self.__tablet_client.execute_remote('ps -A | grep ffmpeg', allowable_code=1, read_output=True)
            return bool(output)  # equal to True if not None and not empty
        except Exception as e:
            self.__logger.warning('Can\'t get screen recorder status: %s', e)
            return False

    def download_last_recording(self, local_path: str):
        self.__tablet_client.download_file(self.last_processed_path,
                                           self.last_processed_file,
                                           local_path)
