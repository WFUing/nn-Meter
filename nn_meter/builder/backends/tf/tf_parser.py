from nn_meter.builder.backends import BaseParser

from nn_meter.builder.backend_meta.utils import ProfiledResults, Latency

class TFParser(BaseParser):
    def __init__(self):
        self.latency = Latency()

    def parse(self, content):
        # 假设内容格式为 "Average latency: 12.34ms"
        avg_latency = float(content.split(":")[1].strip("ms"))
        self.latency = Latency(avg_latency)
        return self

    @property
    def results(self):
        return ProfiledResults({'latency': self.latency})