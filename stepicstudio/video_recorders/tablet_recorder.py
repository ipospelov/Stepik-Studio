import logging

from django.conf import settings
from singleton_decorator import singleton

from stepicstudio import const
from stepicstudio.operationsstatuses.operation_result import InternalOperationResult
from stepicstudio.operationsstatuses.statuses import ExecutionStatus
from stepicstudio.ssh_connections.tablet_client import TabletClient
from stepicstudio.video_recorders.postprocessable_recorder import PostprocessableRecorder


@singleton
class TabletScreenRecorder(PostprocessableRecorder):
    def __init__(self):
        super().__init__()
        self.__command = settings.FFMPEG_TABLET_CMD
        self.__logger = logging.getLogger('stepic_studio.video_recorders.tablet_recorder')
        self.__last_processed_path = None
        self.__last_processed_file = None
        self._load_postprocessing_pipe(settings.TABLET_POSTPROCESSING_PIPE)

        try:
            self.__tablet_client = TabletClient()
        except Exception as e:
            self.__logger.error('Can\'t initialize tablet client: %s', e)
            raise e

    def start_recording(self, path: str, filename: str) -> InternalOperationResult:
        if self.is_active():
            self.__logger.error('Can\'t start FFMPEG for file %s: screen is acctually recording ',
                                path + '/' + filename)
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR, 'Tablet is actually recording')

        self.__tablet_client.check_and_create_folder(path)
        command = settings.FFMPEG_TABLET_CMD + path + '/' + filename + const.SUBSTEP_SCREEN + ' 2< /dev/null &'

        try:
            self.__tablet_client.execute_remote(command)
        except Exception as e:
            self.__logger.error('Screen recording start failed: %s; FFMPEG command: %s', e, command)
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR)

        self.__last_processed_file = filename
        self.__last_processed_path = path
        self.__logger.info('Successfully start screen recording (FFMPEG command: %s', command)
        return InternalOperationResult(ExecutionStatus.SUCCESS)

    def stop_recording(self) -> InternalOperationResult:
        if not self.is_active():
            self.__logger.error('Tablet screencast isn\'t active: can\'t stop non existing FFMPEG process')
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR)

        command = 'pkill -f ffmpeg'

        try:
            #  read_output=True is using for synchronized execution
            self.__tablet_client.execute_remote(command, allowable_code=1, read_output=True)
        except Exception as e:
            self.__logger.error('Problems while stop screen recording: %s', e)
            return InternalOperationResult(ExecutionStatus.FATAL_ERROR)

        self._apply_pipe(self.__last_processed_path, self.__last_processed_file)
        self.__logger.info('Successfully stop screen recording.')
        return InternalOperationResult(ExecutionStatus.SUCCESS)

    def is_active(self) -> bool:
        try:
            output = self.__tablet_client.execute_remote('ps -A | grep ffmpeg', allowable_code=1, read_output=True)
            return bool(output)  # equal to True if not None and not empty
        except Exception as e:
            self.__logger.warning('Can\'t get screen recorder status: %s', e)
            return False