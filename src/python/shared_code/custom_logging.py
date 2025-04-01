

def color_logging_text(message: str, color: str) -> str:
    """Change the color of the logging message."""
    color_map = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'purple': '35',
        'cyan': '36',
        'white': '37'
    }
    
    color_code = color_map.get(color, '37')
    return f"\033[{color_code}m{message}\033[0m"