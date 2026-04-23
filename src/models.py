from pydantic import BaseModel


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = ""


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict | None = None
    delta: dict | None = None
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: int | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail
