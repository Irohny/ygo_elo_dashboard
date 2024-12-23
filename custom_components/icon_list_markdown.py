def icon_list(feature_name: str, symbol: str, amount: int, pos: str):
    icon_string = "".join(f":{symbol}: " for _ in range(amount))
    pos.markdown(f"{feature_name:<15} " + icon_string)
