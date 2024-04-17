_handlers = {}


def register_handler(handler_name, handler_class):
    _handlers[handler_name] = handler_class


def get_handler(asset_info):
    handler_name = asset_info.get("handler")
    try:
        handler_class = _handlers[handler_name]
    except KeyError:
        raise KeyError(f"Handler {handler_name} not available. Ensure it's correct.")
    return handler_class
