from pydantic import BaseModel

class RewriteResult(BaseModel):
    final_prompt: str
    was_rewritten: bool
    original_prompt: str