from exec_comp.models.base import KeypointNet


class ForwardKeypointModel(KeypointNet):
    def __init__(self,chunk=100):
        super().__init__(chunk=chunk,forward=True)
