"""The options the admin page can change. Each can still start from its environment variable; a saved value wins."""

from settings import Setting

SETTINGS = [
    Setting("update_check_hours", "Check for updates every (hours)", "float", 6.0, "SERVICE_UPDATE_CHECK_HOURS", 0, 720,
            help="0 = only when you press Check now"),
    Setting("max_inflight", "Calls allowed to wait for the model", "int", 0, "SERVICE_MAX_INFLIGHT", 0, 10000,
            help="Per replica; more get 429. 0 = 4"),
    Setting("cache_size", "Answers kept per replica", "int", 256, "SERVICE_CACHE_SIZE", 0, 100000,
            help="The last answers for the same request, sent again without scoring. Decisions are deterministic, so every request is kept. 0 = off"),
    Setting("warm", "Load the model at start", "bool", True, "SERVICE_WARM", restart=True),
]
