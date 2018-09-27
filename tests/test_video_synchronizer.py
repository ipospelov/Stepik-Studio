import os
from django.test import TestCase

from stepicstudio.postprocessing.video_sync import VideosSynchronizer

os.environ['DJANGO_SETTINGS_MODULE'] = 'STEPIC_STUDIO.settings'


class TestReencodeMethod(TestCase):
    def test_delay_computing(self):
        video_1 = 'D:\\STEPIKSTUDIO\\TESTER\\test_course\\new_\\step\\Step2from131\\Step2from131_Professor.mp4'
        video_2 = 'D:\\STEPIKSTUDIO\\TESTER\\test_course\\new_\\step\\Step2from131\\Step2from131_Screen.mp4'
        synchronizer = VideosSynchronizer()
        latencies, dropouts = synchronizer.synchronize(video_2, video_1)
        print(latencies)
        print(dropouts)
