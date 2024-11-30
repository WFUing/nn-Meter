from nn_meter.builder.backends import BaseBackend


class DebugBackend(BaseBackend):
    """ For debug use when there is no backend available. All latency value are randomly generated.
    """

    def profile(self, converted_model, metrics=['latency'], input_shape=None, **kwargs):
        import random
        from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults
        latency = Latency(random.randrange(0, 10000) / 100, random.randrange(0, 1000) / 1000)
        return ProfiledResults({'latency': latency}).get(metrics)

    def test_connection(self):
        """ check the status of backend interface connection.
        """
        import logging
        logging.info("hello backend !")