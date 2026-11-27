"""TorchServe handler for the MalGraph server."""
import torch
from ts.torch_handler.base_handler import BaseHandler

from classifier.malgraph.server import MalGraphServer


class MalGraphHandler(BaseHandler):
    def initialize(self, context):
        props = context.system_properties
        if torch.cuda.is_available() and props.get('gpu_id') is not None:
            device = 'cuda:' + str(props.get('gpu_id'))
        else:
            device = 'cpu'
        self.model = MalGraphServer(device=device)
        self.initialized = True

    def preprocess(self, data):
        for row in data:
            bytez = row.get('bytez')
            return bytes(bytez)

    def inference(self, data, *args, **kwargs):
        return self.model.score_pe(data)

    def postprocess(self, data):
        # treat an unparseable graph as malicious (score = 1.0)
        return [data if data is not None else 1.0]
