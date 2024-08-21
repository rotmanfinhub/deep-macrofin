class EventHandler(object):
    def __init__(self) -> None:
        self.event_handlers = []
    
    def __iadd__(self, evt):
        '''Add an event function by += operator'''
        self.event_handlers.append(evt)
        return self
    
    def __isub__(self, evt):
        '''Remove an event function by -= operator'''
        if evt in self.event_handlers:
            self.event_handlers.remove(evt)
        return self
    
    def clear(self):
        '''Remove all event functions'''
        self.event_handlers.clear()
    
    def __call__(self, *args, **kwargs):
        for evt in self.event_handlers:
            evt(*args, **kwargs)