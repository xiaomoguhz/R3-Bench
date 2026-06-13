

import os

from .utils.py_functional import is_package_available


if is_package_available("modelscope"):
    from modelscope.utils.hf_util import patch_hub  # type: ignore


__version__ = "0.3.2.dev0"


if os.getenv("USE_MODELSCOPE_HUB", "0").lower() in ["true", "y", "1"]:
    # Patch hub to download models from modelscope to speed up.
    if not is_package_available("modelscope"):
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope`.")

    patch_hub()
