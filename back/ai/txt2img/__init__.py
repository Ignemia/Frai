"""Alias package pointing to the text2img implementation."""

from importlib import import_module

_impl = import_module('..text2img', __name__)

from ..text2img import *  # noqa: F401,F403

preprocessing = import_module('..text2img.preprocessing', __name__)
utils = import_module('..text2img.utils', __name__)
postprocessing = import_module('..text2img.postprocessing', __name__)

import sys
sys.modules[__name__ + '.preprocessing'] = preprocessing
sys.modules[__name__ + '.utils'] = utils
sys.modules[__name__ + '.postprocessing'] = postprocessing

initialize_txt2img_system = _impl.initialize_txt2img_system
get_txt2img_ai_instance = _impl.get_txt2img_ai_instance
generate_image_from_text = _impl.generate_image_from_text

__all__ = ['preprocessing', 'utils', 'postprocessing',
           'initialize_txt2img_system', 'get_txt2img_ai_instance', 'generate_image_from_text']
