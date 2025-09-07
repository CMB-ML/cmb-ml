_handlers = {}


def register_handler(handler_name, handler_class):
    _handlers[handler_name] = handler_class


def get_handler(asset_info, source_stage=None):
    handler_name = asset_info.get("handler")
    try:
        handler_class = _handlers[handler_name]
    except KeyError:
        src_str = ""
        if source_stage is not None:
            src_str = f" within {source_stage}"
        raise KeyError(f"Handler {handler_name} not available{src_str}. Ensure it's correct and properly registered.")
    return handler_class
