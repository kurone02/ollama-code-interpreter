from typing import Any, Iterator, Mapping, Optional
from jupyter import JupyterNotebook
import ollama
import prompts
from utils import extract_code_blocks

class CodeInterpreter:
    def __init__(self, model: str="", few_shot: bool=True) -> None:
        self.notebook: JupyterNotebook = JupyterNotebook()
        self.model: str = model
        self.dialog: list[dict[str, str]] = []
        self.dialog.extend(prompts.system)
        if few_shot:
            self.dialog.extend(prompts.few_shot_1)
            self.dialog.extend(prompts.few_shot_2)
            self.dialog.extend(prompts.few_shot_3)
            self.dialog.extend(prompts.few_shot_4)

    def reset_dialog(self, few_shot: bool=True):
        self.dialog = []
        self.dialog.extend(prompts.system)
        if few_shot:
            self.dialog.extend(prompts.few_shot_1)
            self.dialog.extend(prompts.few_shot_2)
            self.dialog.extend(prompts.few_shot_3)
            self.dialog.extend(prompts.few_shot_4)

    def _add_dialog_content(self, role: str, content: str) -> None:
        self.dialog.append(
            {
                "role": role,
                "content": content,
            }
        )

    def add_user_content(self, content: str) -> None:
        self._add_dialog_content(role="user", content=content)

    def add_assistant_content(self, content: str) -> None:
        self._add_dialog_content(role="assistant", content=content)

    def generate(self, options: Optional[dict[str, Any]]=None) -> (Mapping[str, Any] | Iterator[Mapping[str, Any]]):
        response = ollama.chat(model=self.model, messages=self.dialog, options=options)
        return response

    def interpret(self, request: str, options: Optional[dict[str, Any]]=None, max_attempts: int=5, verbose: bool=False) -> list[str]:
        results = []
        self.add_user_content(request)
        for attempt in range(max_attempts):
            response = self.generate(options)
            generated_text = response["message"]["content"]
            has_code, code_blocks = extract_code_blocks(generated_text)

            if not has_code:
                results.append(generated_text)
                break

            code = code_blocks[0]

            first_code_pos = generated_text.find(code)
            generated_text = generated_text[:first_code_pos]

            output, error_flag = self.notebook.add_and_run(code)

            if verbose:
                print(f"""```RESULT\n{output}\n```""")

            answer = f"""{generated_text}\n{code}\n```\n```RESULT\n{output}\n```\n"""
            results.append(answer)

            if verbose:
                print(answer)

            self.add_assistant_content(answer)
            self.add_user_content("Keep going.\n")

        return results
    
    def close(self) -> None:
        self.notebook.close()


if __name__ == "__main__":
    MODEL_ID = "deepseek-coder:33b-instruct-q8_0"
    interpreter = CodeInterpreter(model=MODEL_ID)

    responses = interpreter.interpret("What is the 69-th value of the Catalan sequence?", options={
        "temperature": 0.8,
    })

    for response in responses:
        print(response)

    interpreter.close()