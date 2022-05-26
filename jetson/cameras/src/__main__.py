from rover_msgs import Cameras
import lcm

import sys
sys.path.insert(0, "/usr/lib/python3.6/dist-packages")  # 3.6 vs 3.8
import jetson.utils  # noqa

__lcm: lcm.LCM
__pipelines = [None] * 4

ARGUMENTS_LOW = ['--headless', '--bitrate=300000', '--width=256', '--height=144']
# 10.0.0.1 represents the ip of the main base station laptop
# 10.0.0.2 represents the ip of the secondary science laptop
remote_ip = ["10.0.0.1:5000", "10.0.0.1:5001", "10.0.0.2:5000", "10.0.0.2:5001"]
video_sources = [None] * 10


class Pipeline:
    def __init__(self, port):
        self.video_source = None
        self.video_output = jetson.utils.videoOutput(f"rtp://{remote_ip[port]}", argv=ARGUMENTS_LOW)
        self.device_number = -1
        self.port = port

    def update(self):
        image = self.video_source.Capture()
        self.video_output.Render(image)

    def is_open(self):
        return True

    def get_device_number(self):
        return self.device_number

    def port(self):
        return self.port

    def update_device_number(self, index):
        self.device_number = index
        if index != -1:
            self.video_source = video_sources[index]
            if self.video_source is not None:
                self.video_output = jetson.utils.videoOutput(f"rtp://{remote_ip[self.port]}", argv=ARGUMENTS_LOW)
            else:
                print(f"Unable to play camera {index} on {remote_ip[self.port]}.")
                self.device_number = -1
        else:
            self.video_source = None

    def is_currently_streaming(self):
        return self.device_number != -1


def start_pipeline(index, port):
    global __pipelines
    try:
        __pipelines[port].update_device_number(index)
        print(f"Playing camera {index} on {remote_ip[port]}.")
    except Exception:
        pass


def close_video_source(index):
    global video_sources
    video_sources[index] = None


def create_video_source(index):
    global video_sources
    if index == -1:
        return
    if video_sources[index] is not None:
        return
    try:
        video_sources[index] = jetson.utils.videoSource(f"/dev/video{index}", argv=ARGUMENTS_LOW)
    except Exception:
        pass


def device_is_not_being_used_by_other_pipelines(excluded_pipeline, device_number):
    # This function checks whether excluded_pipeline is the only pipeline streaming device device_number
    global __pipelines
    # check if any of the other pipelines are using the current device
    for pipeline_number, pipeline in enumerate(__pipelines):
        if pipeline_number == excluded_pipeline:
            continue
        if pipeline.get_device_number() == device_number:
            return False
    return True


def camera_callback(channel, msg):
    global __pipelines

    camera_cmd = Cameras.decode(msg)

    port_devices = camera_cmd.port

    for port_number, requested_port_device in enumerate(port_devices):
        current_device_number = __pipelines[port_number].get_device_number()

        if current_device_number == requested_port_device:
            continue

        # check if we need to close current video source or not
        if device_is_not_being_used_by_other_pipelines(port_number, current_device_number):
            close_video_source(current_device_number)

        create_video_source(requested_port_device)
        start_pipeline(requested_port_device, port_number)


def main():
    global __pipelines, __lcm

    __pipelines = [Pipeline(0), Pipeline(1), Pipeline(2), Pipeline(3)]

    __lcm = lcm.LCM()
    __lcm.subscribe("/cameras_cmd", camera_callback)

    while True:
        while __lcm.handle_timeout(0):
            pass
        for port_number, pipeline in enumerate(__pipelines):
            if pipeline.is_currently_streaming():
                pipeline.update()


if __name__ == "__main__":
    main()