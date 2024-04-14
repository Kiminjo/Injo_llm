# IO 
from typing import Dict

def fill_prompt(template: str, **kwargs: Dict) -> str:
    """
    Fill the template with the given keyword arguments.
    
    Args:
    - template (str): The template to fill.
    - kwargs (dict): The keyword arguments to fill the template.
    
    Returns:
    - str: The filled template.
    """
    return template.format(**kwargs)