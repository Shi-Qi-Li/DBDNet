class Registry(dict):
    def __init__(self, registry_name, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        self._dict = {}
        self.name = registry_name

    def __call__(self, target):
        return self.register(target)

    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")
            if key in self._dict:
                print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
            self[key] = value
            return value

        if callable(target):
            return add_item(target.__name__, target)
        else: 
            return lambda x : add_item(target, x)
    
    def build(self, cfg):
        obj_name = cfg.pop("name")
        if isinstance(obj_name, str):
            obj_cls = self.__getitem__(obj_name)
            if obj_cls is None:
                raise KeyError(f'{obj_name} is not in the {self.name} registry')
        else:
            raise TypeError(f'Name must be a str, but got {type(obj_name)}')
        
        try:
            return obj_cls(**cfg)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')


    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()