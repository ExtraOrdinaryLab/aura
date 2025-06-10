import re
from typing import List, Union

from zhconv import convert


# Remove punctuation from a string or list of strings
def remove_punctuation(input_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Removes specified punctuation characters from a string or a list of strings.
    
    Args:
        input_text (Union[str, List[str]]): The input text or list of texts to process.
        
    Returns:
        Union[str, List[str]]: Text(s) with punctuation removed.
    
    Raises:
        Exception: If the input type is not supported.
    """
    punctuation_chars = '!,.;:?、！，。；：？'
    if isinstance(input_text, str):
        # Remove punctuation from a single string
        processed_text = re.sub(rf"[{punctuation_chars}]+", "", input_text).strip()
        return processed_text
    elif isinstance(input_text, list):
        # Remove punctuation from each string in the list
        processed_text_list = [
            re.sub(rf"[{punctuation_chars}]+", "", text).strip() for text in input_text
        ]
        return processed_text_list
    else:
        raise Exception(f"Unsupported input type: {type(input_text)}")


# Convert Traditional Chinese text to Simplified Chinese
def convert_to_simplified_chinese(input_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Converts Traditional Chinese text to Simplified Chinese.
    
    Args:
        input_text (Union[str, List[str]]): The input text or list of texts to process.
        
    Returns:
        Union[str, List[str]]: Text(s) converted to Simplified Chinese.
    
    Raises:
        Exception: If the input type is not supported.
    """
    if isinstance(input_text, str):
        # Convert a single string to Simplified Chinese
        simplified_text = convert(input_text, 'zh-cn')
        return simplified_text
    elif isinstance(input_text, list):
        # Convert each string in the list to Simplified Chinese
        simplified_text_list = [convert(text, 'zh-cn') for text in input_text]
        return simplified_text_list
    else:
        raise Exception(f"Unsupported input type: {type(input_text)}")


def remove_punctuation_sapc(text: str) -> str:
    """
    Removes punctuation from the input text while retaining specific Unicode characters.

    Args:
        text (str): The input string to process.

    Returns:
        str: The processed string with punctuation removed.
    """
    # Define the Unicode character set to retain
    allowed_unicode_codes = (
        "\u0041-\u005a"  # Uppercase A-Z
        "\u0027"         # Apostrophe (')
        "\u0020"         # Space
        "\u00c0\u00c1\u00c4\u00c5"  # À, Á, Ä, Å
        "\u00c8\u00c9\u00cd\u00cf"  # È, É, Í, Ï
        "\u00d1\u00d3\u00d6\u00d8"  # Ñ, Ó, Ö, Ø
        "\u00db\u00dc\u0106"        # Û, Ü, Ć
    )

    # Process the text: strip, convert to uppercase, and remove unwanted characters
    text = text.strip().upper()
    text = re.sub(rf"[^{allowed_unicode_codes}]", "", text)
    text = " ".join(text.split())  # Normalize spaces

    return text
