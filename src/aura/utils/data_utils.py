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
