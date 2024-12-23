def is_player(user_roles: list) -> bool:
    return any([role in ["player", "admin"] for role in user_roles])


def is_admin(user_roles: list) -> bool:
    return "admin" in user_roles


def is_reporter(user_roles: list) -> bool:
    return any([role in ["player", "admin"] for role in user_roles])
