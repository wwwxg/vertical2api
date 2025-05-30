import json
import time
import uuid
import threading
import hashlib
from collections import OrderedDict
from typing import Any, AsyncGenerator, Dict, List, Optional
import os 
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from vertical_client import VerticalApiClient, USER_AGENT

# Configuration
CONVERSATION_CACHE_MAX_SIZE = 100
DEFAULT_REQUEST_TIMEOUT = 30.0
CLEAR_CHAT_AFTER_RESPONSE = os.getenv("CLEAR_CHAT_AFTER_RESPONSE", "false").lower() == "true"

# Global variables
VALID_CLIENT_KEYS: set = set()
VERTICAL_AUTH_TOKENS: list = []
current_vertical_token_index: int = 0
token_rotation_lock = threading.Lock()
models_data: Dict[str, Any] = {}
http_client: Optional[httpx.AsyncClient] = None
vertical_api_client: Optional[VerticalApiClient] = None
conversation_cache: OrderedDict = OrderedDict()
cache_lock = threading.Lock()

# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None

class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
    usage: Optional[Dict[str, int]] = None # Added for OpenAI spec compliance in final chunk

# FastAPI App
app = FastAPI(title="Vertical OpenAI Compatible API")
security = HTTPBearer(auto_error=False)

# Helper functions
def generate_message_fingerprint(role: str, content: str) -> str:
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    return f"{role}:{content_hash}"

def load_models():
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        if "data" in raw_data:
            processed_data = raw_data
        elif "models" in raw_data:
            processed_models = []
            for model in raw_data["models"]:
                model_entry = {
                    "id": model.get("modelId", ""), "object": "model",
                    "created": int(time.time()), "owned_by": "vertical-studio",
                    "vertical_model_id": model.get("modelId", ""),
                    "vertical_model_url": model.get("url", "")
                }
                thinking_entry = model_entry.copy()
                thinking_entry["id"] = f"{model_entry['id']}-thinking"
                thinking_entry["description"] = f"{model_entry['id']} (with thinking steps)"
                model_entry["description"] = f"{model_entry['id']} (final answer only)"
                processed_models.extend([model_entry, thinking_entry])
            processed_data = {"data": processed_models}
        else:
            processed_data = {"data": []}
            
        for model in processed_data["data"]:
            model["output_reasoning_flag"] = model["id"].endswith("-thinking")
            if model.get("created", 0) == 0: model["created"] = int(time.time())
        return processed_data
    except Exception as e:
        print(f"ERROR loading models.json: {e}")
        return {"data": []}

def load_client_api_keys():
    global VALID_CLIENT_KEYS
    try:
        # 尝试从环境变量读取
        env_keys = os.getenv("CLIENT_API_KEYS")
        if env_keys:
            loaded_keys = [key.strip() for key in env_keys.split(",")]
            VALID_CLIENT_KEYS = set(loaded_keys)
            print(f"Loaded {len(VALID_CLIENT_KEYS)} client API key(s) from environment variables.")
            return

        # 如果环境变量没有设置，则从文件读取
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            if not isinstance(keys, list):
                print("WARNING: client_api_keys.json should contain a list of keys.")
                VALID_CLIENT_KEYS = set()
                return
            VALID_CLIENT_KEYS = set(keys)
            if not VALID_CLIENT_KEYS:
                print("WARNING: client_api_keys.json is empty.")
            else:
                print(f"Loaded {len(VALID_CLIENT_KEYS)} client API key(s).")
    except FileNotFoundError:
        print("ERROR: client_api_keys.json not found.")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"ERROR loading client_api_keys.json: {e}")
        VALID_CLIENT_KEYS = set()

def load_vertical_auth_tokens():
    global VERTICAL_AUTH_TOKENS
    try:
        # 尝试从环境变量读取
        env_tokens = os.getenv("VERTICAL_AUTH_TOKENS")
        if env_tokens:
            loaded_tokens = [parse_token_line(token.strip()) for token in env_tokens.split(",")]
            VERTICAL_AUTH_TOKENS = loaded_tokens
            print(f"Loaded {len(VERTICAL_AUTH_TOKENS)} Vertical auth token(s) from environment variables.")
            return

        # 如果环境变量没有设置，则从文件读取
        with open("vertical.txt", "r", encoding="utf-8") as f: lines = f.readlines()
        loaded_tokens = [parse_token_line(line.strip()) for line in lines if line.strip()]
        VERTICAL_AUTH_TOKENS = loaded_tokens
        if not VERTICAL_AUTH_TOKENS:
            print("WARNING: No valid tokens found in vertical.txt.")
        else:
            print(f"Loaded {len(VERTICAL_AUTH_TOKENS)} Vertical auth token(s) from file.")
    except FileNotFoundError:
        print("ERROR: vertical.txt not found.")
        VERTICAL_AUTH_TOKENS = []
    except Exception as e:
        print(f"ERROR loading vertical.txt: {e}")
        VERTICAL_AUTH_TOKENS = []

def parse_token_line(line: str) -> dict:
    parts = line.split("----")
    if len(parts) == 1:
        print(f"[DEBUG] Token line has only token: {parts[0]}")  # <<< 新增日志
        return {"token": parts[0], "email": None, "password": None}
    elif len(parts) >= 3:
        print(f"[DEBUG] Loaded token with email & password: {parts[0]} ---- {parts[1]}")  # <<< 新增日志
        return {
            "token": parts[0],
            "email": parts[1],
            "password": parts[2]
        }
    else:
        raise ValueError(f"Invalid token line format: {line}")

async def refresh_auth_token(email: str, password: str) -> Optional[str]:
    """
    使用 email 和 password 自动登录并获取新的 auth-token。
    """
    print(f"[DEBUG] 正在尝试自动登录：{email}")
    async with httpx.AsyncClient() as client:
        try:
            # Step 1: 发送邮箱
            resp1 = await client.post("https://app.verticalstudio.ai/login.data", data={"email": email})
            if resp1.status_code not in [200, 202]:
                print(f"[ERROR] Failed to send email during login: {resp1.status_code}")
                return None

            # Step 2: 获取下一步URL并提交密码
            if resp1.status_code == 202:
                location = resp1.headers.get("location")
                if location:
                    # 正确的URL构造：在query参数前插入.data
                    if "?" in location:
                        base_path, query_params = location.split("?", 1)
                        login_url = f"https://app.verticalstudio.ai{base_path}.data?{query_params}"
                    else:
                        login_url = f"https://app.verticalstudio.ai{location}.data"
                    print(f"[DEBUG] Using location from 202 response: {login_url}")
                else:
                    print(f"[ERROR] No location header in 202 response")
                    return None
            else:
                # 原有逻辑保持不变（适用于200响应）
                login_url = f"https://app.verticalstudio.ai/login-password.data?email={httpx.URL(resp1.json()[1]).params['email']}"
                print(f"[DEBUG] Using original logic for 200 response: {login_url}")
            
            resp2 = await client.post(login_url, data={"email": email, "password": password})

            # ✅ 关键修复：202状态码是成功的！
            if resp2.status_code not in [200, 202]:
                print(f"[ERROR] Failed to login with email {email}: {resp2.status_code}")
                return None

            auth_token = resp2.cookies.get("sb-ppdjlmajmpcqpkdmnzfd-auth-token")
            if not auth_token:
                print(f"[ERROR] No auth-token found in response cookies for {email}.")
                print(f"[DEBUG] Response status: {resp2.status_code}, cookies: {dict(resp2.cookies)}")
                return None

            print(f"[INFO] Successfully refreshed auth-token for {email}.")
            return auth_token

        except Exception as e:
            print(f"[ERROR] Exception during token refresh: {e}")
            return None

def get_model_item(model_id: str) -> Optional[Dict]:
    return next((model for model in models_data.get("data", []) if model.get("id") == model_id), None)

async def authenticate_client(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not VALID_CLIENT_KEYS: raise HTTPException(status_code=503, detail="Service unavailable: Client API keys not configured.")
    if not auth or not auth.credentials:
        raise HTTPException(status_code=401, detail="API key required in Authorization header.", headers={"WWW-Authenticate": "Bearer"})
    if auth.credentials not in VALID_CLIENT_KEYS: raise HTTPException(status_code=403, detail="Invalid client API key.")

async def get_next_vertical_auth_token(failed_email: str = None) -> str:
    global current_vertical_token_index

    # 优先刷新失败的账户
    if failed_email:
        for account in VERTICAL_AUTH_TOKENS:
            if account["email"] == failed_email and account["password"]:
                print(f"[DEBUG] Refreshing failed account: {failed_email}")
                new_token = await refresh_auth_token(account["email"], account["password"])
                if new_token:
                    account["token"] = new_token
                    print(f"[INFO] Successfully refreshed token for account: {failed_email}")
                    return new_token

    # 正常轮换机制
    attempts = 0
    max_attempts = len(VERTICAL_AUTH_TOKENS)
    
    while attempts < max_attempts:
        with token_rotation_lock:
            account = VERTICAL_AUTH_TOKENS[current_vertical_token_index]
            account_index = current_vertical_token_index
            current_vertical_token_index = (current_vertical_token_index + 1) % len(VERTICAL_AUTH_TOKENS)

        if account["token"]:
            print(f"[DEBUG] Using existing token for account index {account_index}, 邮箱: {account.get('email', 'N/A')}")
            return account["token"]

        print(f"[DEBUG] Token empty for account index {account_index}. Attempting to refresh this account...")
        if account["email"] and account["password"]:
            new_token = await refresh_auth_token(account["email"], account["password"])
            if new_token:
                account["token"] = new_token
                print(f"[INFO] Successfully refreshed token for account index {account_index}.")
                return new_token
            else:
                print(f"[ERROR] Failed to refresh token for account index {account_index}.")
        else:
            print(f"[ERROR] Account index {account_index} missing email or password for refresh.")
        
        attempts += 1

    # 最终兜底：尝试刷新第一个可用凭证的账户
    for account in VERTICAL_AUTH_TOKENS:
        if account["email"] and account["password"]:
            print(f"[DEBUG] Trying fallback refresh on account: {account['email']}")
            new_token = await refresh_auth_token(account["email"], account["password"])
            if new_token:
                account["token"] = new_token
                print(f"[INFO] Successfully refreshed token via fallback for account: {account['email']}")
                return new_token

    raise HTTPException(status_code=503, detail="All auth tokens failed and no valid credentials available for refresh.")

@app.on_event("startup")
async def startup():
    global models_data, http_client, vertical_api_client
    models_data = load_models()
    load_client_api_keys()
    load_vertical_auth_tokens()
    http_client = httpx.AsyncClient(timeout=None)
    vertical_api_client = VerticalApiClient(http_client)
    print("Vertical OpenAI Compatible API server started.")

@app.on_event("shutdown")
async def shutdown():
    if http_client: await http_client.aclose()
    print("Vertical OpenAI Compatible API server shut down.")

@app.get("/v1/models", response_model=ModelList)
async def list_models(_: None = Depends(authenticate_client)):
    return ModelList(data=[ModelInfo(
        id=model.get("id", ""), created=model.get("created", int(time.time())),
        owned_by=model.get("owned_by", "vertical-studio")
    ) for model in models_data.get("data", [])])

def parse_json_string_content(line: str, prefix_len: int, suffix_len: int) -> str:
    content_segment = line[prefix_len:suffix_len]
    try:
        return json.loads(f'"{content_segment}"')
    except json.JSONDecodeError:
        # print(f"Warning: JSONDecodeError in parse_json_string_content, fallback. Raw segment: {content_segment[:100]}...")
        return content_segment.replace('\\\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
    except Exception as e:
        print(f"Error: Unexpected error in parse_json_string_content: {e}. Raw segment: {content_segment[:100]}...")
        return content_segment

def _update_conversation_cache(
    is_new_cached_conv: bool, vertical_chat_id_for_cache: str,
    matched_conv_id_for_cache_update: Optional[str],
    original_request_messages: List[ChatMessage], full_assistant_reply_str: str,
    system_prompt_hash_for_cache: int, model_url_for_cache: str
):
    with cache_lock:
        if is_new_cached_conv:
            new_internal_id = str(uuid.uuid4())
            current_fingerprints = [generate_message_fingerprint(msg.role, msg.content) for msg in original_request_messages]
            current_fingerprints.append(generate_message_fingerprint("assistant", full_assistant_reply_str))
            conversation_cache[new_internal_id] = {
                "vertical_chat_id": vertical_chat_id_for_cache, "vertical_model_url": model_url_for_cache,
                "system_prompt_hash": system_prompt_hash_for_cache,
                "message_fingerprints": current_fingerprints, "last_seen": time.time()
            }
            if len(conversation_cache) > CONVERSATION_CACHE_MAX_SIZE: conversation_cache.popitem(last=False)
        elif matched_conv_id_for_cache_update:
            cached_item = conversation_cache[matched_conv_id_for_cache_update]
            if original_request_messages:
                last_user_msg = original_request_messages[-1]
                last_user_fingerprint = generate_message_fingerprint(last_user_msg.role, last_user_msg.content)
                if not cached_item["message_fingerprints"] or cached_item["message_fingerprints"][-1] != last_user_fingerprint:
                    cached_item["message_fingerprints"].append(last_user_fingerprint)
            cached_item["message_fingerprints"].append(generate_message_fingerprint("assistant", full_assistant_reply_str))
            cached_item["last_seen"] = time.time()

async def openai_stream_adapter(
    api_stream_generator: AsyncGenerator[str, None], model_name_for_response: str,
    reasoning_requested: bool, vertical_chat_id_for_cache: str, is_new_cached_conv: bool,
    matched_conv_id_for_cache_update: Optional[str], original_request_messages: List[ChatMessage],
    system_prompt_hash_for_cache: int, model_url_for_cache: str
) -> AsyncGenerator[str, None]:
    full_assistant_reply_parts = []
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    first_chunk_sent = False
    
    try:
        # print("[ADAPTER] Starting openai_stream_adapter...")
        async for line in api_stream_generator:
            line_stripped = line.strip()
            # print(f"[ADAPTER] Raw line from generator: '{line_stripped}'") # Too verbose for general use

            if line_stripped.startswith("error:"):
                try: error_data = json.loads(line_stripped[6:])
                except: error_msg = "Unknown error from Vertical API"
                else: error_msg = error_data.get("message", "Unknown error")
                print(f"[ADAPTER] Detected error line: {line_stripped}, message: {error_msg}")
                error_resp = StreamResponse(
                    id=stream_id, model=model_name_for_response,
                    choices=[StreamChoice(delta={"role": "assistant", "content": f"Error: {error_msg}"}, index=0, finish_reason="stop")]
                )
                yield f"data: {error_resp.model_dump_json(exclude_none=True)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            delta_payload = None
            if line_stripped.startswith('0:"') and line_stripped.endswith('"'):
                final_content = parse_json_string_content(line_stripped, 3, -1)
                # print(f"[ADAPTER] Parsed 0: content: '{final_content[:50]}...'")
                delta_payload = {"role": "assistant", "content": final_content} if not first_chunk_sent else {"content": final_content}
                full_assistant_reply_parts.append(final_content)
            elif reasoning_requested and line_stripped.startswith('g:"') and line_stripped.endswith('"'):
                thinking_content = parse_json_string_content(line_stripped, 3, -1)
                # print(f"[ADAPTER] Parsed g: content: '{thinking_content[:50]}...'")
                full_assistant_reply_parts.append(f"[Thinking]: {thinking_content}")
                delta_payload = {"role": "assistant", "reasoning_content": thinking_content} if not first_chunk_sent else {"reasoning_content": thinking_content}
            elif line_stripped.startswith('d:'):
                try:
                    event_data = json.loads(line_stripped[2:])
                    # print(f"[ADAPTER] Parsed d: event: {event_data}")

                    if "finishReason" in event_data:
                        actual_finish_reason = event_data.get("finishReason")
                        usage_from_vertical = event_data.get("usage")

                        final_chunk_dict = {
                            "id": stream_id, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": model_name_for_response,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": actual_finish_reason}]
                        }

                        if usage_from_vertical:
                            prompt_tokens = usage_from_vertical.get("promptTokens", 0)
                            completion_tokens = usage_from_vertical.get("completionTokens", 0)
                            final_chunk_dict["usage"] = {
                                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens
                            }
                        
                        yield f"data: {json.dumps(final_chunk_dict)}\n\n"
                        print(f"[ADAPTER] Yielded final chunk with usage from 'd:' event. Usage: {final_chunk_dict.get('usage')}")
                        break 
                except Exception as e_parse_d:
                    print(f"[ADAPTER] Error parsing d: event '{line_stripped}': {e_parse_d}")
                    pass
            
            if delta_payload:
                stream_resp = StreamResponse(
                    id=stream_id, model=model_name_for_response,
                    choices=[StreamChoice(delta=delta_payload, index=0)]
                )
                if not first_chunk_sent: first_chunk_sent = True
                yield f"data: {stream_resp.model_dump_json(exclude_none=True)}\n\n"
            # elif line_stripped and not any(line_stripped.startswith(p) for p in ["error:", "d:", '0:"', 'g:"']):
                # print(f"[ADAPTER] Line not processed into delta_payload: '{line_stripped}'")

        # print(f"[ADAPTER] Finished adapter loop.")
        full_assistant_reply = "\n".join(full_assistant_reply_parts)
        _update_conversation_cache(
            is_new_cached_conv, vertical_chat_id_for_cache, matched_conv_id_for_cache_update,
            original_request_messages, full_assistant_reply, system_prompt_hash_for_cache, model_url_for_cache
        )
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"ERROR in openai_stream_adapter: {e}")
        import traceback
        traceback.print_exc()
        error_resp_exc = StreamResponse(
            id=stream_id, model=model_name_for_response,
            choices=[StreamChoice(delta={"role": "assistant", "content": f"Internal error: {str(e)}"}, index=0, finish_reason="stop")]
        )
        yield f"data: {error_resp_exc.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

async def clear_vertical_chat(chat_id: str, auth_token: str, vertical_model_id: str):
    """
    发送清除聊天记录的请求。
    """
    # 映射完整模型 ID 到简写
    MODEL_SHORT_NAMES = {
        "claude-4-sonnet-20250514": "claude-4-sonnet",
        "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    }
    short_model_name = MODEL_SHORT_NAMES.get(vertical_model_id, vertical_model_id.split("-")[0])  # 默认取前缀

    clear_url = "https://app.verticalstudio.ai/api/chat/archive.data"
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Origin": "https://app.verticalstudio.ai",
        "Referer": f"https://app.verticalstudio.ai/stream/models/{short_model_name}/{chat_id}",
        "Cookie": f"sb-ppdjlmajmpcqpkdmnzfd-auth-token={auth_token}"
    }
    data = {"chat": chat_id}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(clear_url, headers=headers, data=data)
            if response.status_code == 200:
                print(f"[INFO] 聊天记录已成功清除：{chat_id}")
            else:
                print(f"[ERROR] 清除聊天记录失败：{response.status_code} - {response.text[:100]}")
    except Exception as e:
        print(f"[ERROR] 清除聊天记录时发生异常：{e}")


async def aggregate_stream_for_non_stream_response(
    openai_sse_stream: AsyncGenerator[str, None], model_name: str
) -> ChatCompletionResponse:
    content_parts, reasoning_parts = [], []
    final_usage_data: Optional[Dict[str, int]] = None
    final_finish_reason: str = "stop"

    # print("[AGGREGATOR] Starting to aggregate stream...")
    async for sse_line in openai_sse_stream:
        # print(f"[AGGREGATOR] Raw SSE line: '{sse_line.strip()}'") # Too verbose
        if sse_line.startswith("data: ") and sse_line.strip() != "data: [DONE]":
            try:
                json_data_str = sse_line[6:].strip()
                data = json.loads(json_data_str)
                
                if data.get("choices") and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    if "content" in delta: content_parts.append(delta["content"])
                    elif "reasoning_content" in delta: reasoning_parts.append(delta["reasoning_content"])
                    
                    if choice.get("finish_reason"):
                        final_finish_reason = choice["finish_reason"]
                        # print(f"[AGGREGATOR] Extracted finish_reason: {final_finish_reason}")
                
                if "usage" in data and data["usage"] is not None:
                    final_usage_data = data["usage"]
                    # print(f"[AGGREGATOR] Extracted usage data: {final_usage_data}")
            except json.JSONDecodeError: # Ignore malformed JSON
                # print(f"[AGGREGATOR] JSONDecodeError for line '{json_data_str}'")
                pass
            except Exception as ex:
                print(f"[AGGREGATOR] Exception during aggregation for line '{json_data_str}': {ex}")
                pass
    
    # print(f"[AGGREGATOR] Finished aggregation. Content parts: {len(content_parts)}, Reasoning parts: {len(reasoning_parts)}")
    combined_parts = [f"[Thinking]: {part}" for part in reasoning_parts] + content_parts
    full_content = "".join(combined_parts)
    
    usage_to_return = final_usage_data if final_usage_data else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if final_usage_data: # Ensure total_tokens is sum if provided components
        pt = final_usage_data.get("prompt_tokens", 0)
        ct = final_usage_data.get("completion_tokens", 0)
        usage_to_return["prompt_tokens"] = pt
        usage_to_return["completion_tokens"] = ct
        usage_to_return["total_tokens"] = pt + ct

    # print(f"[AGGREGATOR] Final usage: {usage_to_return}, Final finish_reason: {final_finish_reason}")
    return ChatCompletionResponse(
        model=model_name,
        choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=full_content), finish_reason=final_finish_reason)],
        usage=usage_to_return
    )

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(authenticate_client)
):
    model_config = get_model_item(request.model)
    if not model_config: raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    vertical_model_id = model_config.get("vertical_model_id")
    vertical_model_url = model_config.get("vertical_model_url")
    output_reasoning_active = model_config.get("output_reasoning_flag", False)
    
    if not vertical_model_id or not vertical_model_url:
        raise HTTPException(status_code=500, detail="Model configuration incomplete")
    
    current_system_prompt_str = "".join(msg.content + "\n" for msg in request.messages if msg.role == "system").strip()
    latest_user_message_content = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "") # Get last user message
    if not latest_user_message_content and not any(msg.role == "user" for msg in request.messages): # check if any user message exists
         # If only system messages, and no user message, it's often an error or requires specific handling.
         # For now, let's ensure there's *some* user interaction for the flow.
         # If API allows calls with only system prompt, this check might need adjustment.
         # However, most chat APIs expect at least one user message.
         # If request.messages is just one system message, this will make latest_user_message_content empty.
         # We need to check if there was any user message in the history.
        all_user_messages = [msg.content for msg in request.messages if msg.role == "user"]
        if not all_user_messages:
            raise HTTPException(status_code=400, detail="Request must contain at least one user message.")
        # If there are historical user messages, but the *last* one isn't 'user', that's usually fine for a continuation.
        # The logic below for 'message_to_send_to_vertical' handles history.
        # The 'latest_user_message_content' is key for cache matching and sending to Vertical if continuing.

    current_system_prompt_hash = hash(current_system_prompt_str)
    prefix_message_fingerprints = [generate_message_fingerprint(msg.role, msg.content) for msg in request.messages[:-1]]
    
    matched_conv_id, cached_vertical_chat_id = None, None
    with cache_lock:
        for conv_id, cached_data in reversed(list(conversation_cache.items())):
            if (cached_data["vertical_model_url"] == vertical_model_url and
                cached_data["system_prompt_hash"] == current_system_prompt_hash and
                cached_data["message_fingerprints"] == prefix_message_fingerprints):
                matched_conv_id, cached_vertical_chat_id = conv_id, cached_data["vertical_chat_id"]
                conversation_cache.move_to_end(conv_id)
                break
    
    final_vertical_chat_id, message_to_send_to_vertical, is_new_cached_conversation = None, "", False
    
    if cached_vertical_chat_id:
        final_vertical_chat_id = cached_vertical_chat_id
        # Send only the latest user message if continuing a conversation
        # Ensure latest_user_message_content is from the *actual last user message* in the request
        last_message = request.messages[-1]
        if last_message.role == "user":
            message_to_send_to_vertical = last_message.content
        else: # Should not happen if previous checks are good, but as a fallback:
            raise HTTPException(status_code=400, detail="Last message must be from user to continue a conversation.")
        print(f"Reusing cached Vertical chat_id: {final_vertical_chat_id} for model {request.model}")
    else:
        is_new_cached_conversation = True
        if not vertical_api_client: raise HTTPException(status_code=500, detail="Vertical API client not initialized.")
        auth_token = await get_next_vertical_auth_token()
        # 找到当前token对应的账户邮箱
        auth_token = await get_next_vertical_auth_token()
        current_account_email = "Unknown"
        for account in VERTICAL_AUTH_TOKENS:
            if account["token"] == auth_token:
                current_account_email = account.get("email", "Unknown")
                break
    
        new_chat_id = await vertical_api_client.get_chat_id(vertical_model_url, auth_token, current_account_email)
        if not new_chat_id:
            print(f"[INFO] Chat ID 为空，尝试刷新 token。当前使用的token: {auth_token[:20]}...")
            auth_token = await get_next_vertical_auth_token(current_account_email)  # 强制刷新当前账户
            # 重新获取当前账户邮箱信息
            current_account_email = "Unknown"
            for account in VERTICAL_AUTH_TOKENS:
                if account["token"] == auth_token:
                    current_account_email = account.get("email", "Unknown")
                    break
            new_chat_id = await vertical_api_client.get_chat_id(vertical_model_url, auth_token, current_account_email)
            if not new_chat_id:
                raise HTTPException(status_code=500, detail="Failed to get chat_id after token refresh.")
        final_vertical_chat_id = new_chat_id
        print(f"Created new Vertical chat_id: {final_vertical_chat_id} for model {request.model}")
        
        history_parts = []
        for msg in request.messages: # Send full history for new chat
            if msg.role == "user": history_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant": history_parts.append(f"Assistant: {msg.content}")
        message_to_send_to_vertical = "\n".join(history_parts) if history_parts else latest_user_message_content # Fallback

    if not vertical_api_client: raise HTTPException(status_code=500, detail="Vertical API client not initialized.")
    api_stream_generator = vertical_api_client.send_message_stream(
        auth_token, final_vertical_chat_id, message_to_send_to_vertical,
        vertical_model_id, output_reasoning_active, current_system_prompt_str
    )
    
    openai_sse_stream = openai_stream_adapter(
        api_stream_generator, request.model, output_reasoning_active, final_vertical_chat_id,
        is_new_cached_conversation, matched_conv_id, request.messages,
        current_system_prompt_hash, vertical_model_url
    )
    
    if CLEAR_CHAT_AFTER_RESPONSE and final_vertical_chat_id:
        background_tasks.add_task(clear_vertical_chat, final_vertical_chat_id, auth_token, vertical_model_id)

    if request.stream:
        return StreamingResponse(openai_sse_stream, media_type="text/event-stream")
    else:
        return await aggregate_stream_for_non_stream_response(openai_sse_stream, request.model)

if __name__ == "__main__":
    import os
    if not os.path.exists("client_api_keys.json"):
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump(["sk-your-custom-key-here"], f, indent=2)
        print("Created example client_api_keys.json. Please edit it with your key(s).")
    
    print("Starting Vertical OpenAI Compatible API server...")
    print("Endpoints: GET /v1/models, POST /v1/chat/completions")
    print("Use client API key in Authorization header (Bearer sk-your-key).")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
