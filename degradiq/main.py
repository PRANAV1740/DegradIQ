try:
    from api import app  # noqa: F401
except ImportError:
    from .api import app  # noqa: F401
