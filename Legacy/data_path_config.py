from elp_rumble.config.paths import *  # re-export constants

class DataPathConfig:
    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError as e:
            raise AttributeError(name) from e