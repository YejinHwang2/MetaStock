
import yaml
import inspect

class ARGProcessor():
    def __init__(self, setting_file):
        self.setting_file = setting_file
        self.load()

    def load(self):
        with open(self.setting_file) as file:
            self.kwargs = yaml.load(file, Loader=yaml.FullLoader)
    
    def get_args(self, cls):
        cls_kwargs = {
            k: self.kwargs.get(k) for k in inspect.signature(cls.__init__).parameters.keys() 
            if self.kwargs.get(k) is not None
        }
        return cls_kwargs