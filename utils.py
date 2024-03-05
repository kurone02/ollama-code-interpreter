import re

def extract_code_blocks(text: str):
    pattern = r"```(?:python\n)?(.*?)```" 
    code_blocks = re.findall(pattern, text, re.DOTALL)
    has_code = len(code_blocks) > 0
    return has_code, [block.strip() for block in code_blocks]