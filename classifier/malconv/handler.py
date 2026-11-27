"""TorchServe handler for the MalConv server."""
import os

import torch
from ts.torch_handler.base_handler import BaseHandler

from classifier.malconv.server import MalConvServer, _DEFAULT_MODEL_PATH


class MalConvHandler(BaseHandler):
    def initialize(self, context):
        props = context.system_properties
        if torch.cuda.is_available() and props.get('gpu_id') is not None:
            device = 'cuda:' + str(props.get('gpu_id'))
        else:
            device = 'cpu'
        model_path = os.environ.get('MALCONV_MODEL_PATH', _DEFAULT_MODEL_PATH)
        self.model = MalConvServer(model_path=model_path, device=device)
        self.initialized = True

    def preprocess(self, data):
        for row in data:
            bytez = row.get('bytez')
            return bytes(bytez)

    def inference(self, data, *args, **kwargs):
        return self.model.score_pe(data)

    def postprocess(self, data):
        return [data]
