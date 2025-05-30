import httpx
import json
import time
import random
import string
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, Dict, Any
from urllib.parse import urlparse

DEFAULT_REQUEST_TIMEOUT = 30.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

class VerticalApiClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    def _generate_message_id(self, length: int = 16) -> str:
        return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

    def _get_iso_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    async def get_chat_id(self, model_data_url: str, auth_token: str, account_email: str = "Unknown") -> Optional[str]:
        headers = {
            "Cookie": f"sb-ppdjlmajmpcqpkdmnzfd-auth-token={auth_token}",
            "User-Agent": USER_AGENT, "Accept": "text/plain,*/*;q=0.8",
            "Referer": "https://app.verticalstudio.ai/"
        }
        url_with_force_new = f"{model_data_url}?forceNewChat=true"
        
        # print(f"[VCLIENT_DEBUG] get_chat_id: URL: {url_with_force_new}")
        # print(f"[VCLIENT_DEBUG] get_chat_id: Auth (first 30): {auth_token[:30]}...")

        try:
            response = await self.client.get(url_with_force_new, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT)
            # print(f"[VCLIENT_DEBUG] get_chat_id: Response status: {response.status_code}")

            chat_id_to_return = None
            if response.status_code == 202: # Typically indicates chat creation, Location header has the ID
                location_header = response.headers.get("Location")
                if location_header:
                    parsed_url = urlparse(location_header)
                    path_segments = parsed_url.path.strip('/').split('/')
                    potential_chat_id = path_segments[-1] if path_segments else None
                    if potential_chat_id and potential_chat_id.startswith("cmb"):
                        chat_id_to_return = potential_chat_id
                        # print(f"[VCLIENT_DEBUG] get_chat_id: Success from 202 Location: {chat_id_to_return}")
                # else: print(f"[VCLIENT_WARN] get_chat_id: Status 202 but no Location header.")
            elif response.history: # Followed redirects
                initial_redirect_response = response.history[0]
                if initial_redirect_response.is_redirect:
                    location_header_hist = initial_redirect_response.headers.get("Location")
                    if location_header_hist:
                        parsed_url_hist = urlparse(location_header_hist)
                        path_segments_hist = parsed_url_hist.path.strip('/').split('/')
                        potential_chat_id_hist = path_segments_hist[-1] if path_segments_hist else None
                        if potential_chat_id_hist and potential_chat_id_hist.startswith("cmb"):
                            chat_id_to_return = potential_chat_id_hist
                            # print(f"[VCLIENT_DEBUG] get_chat_id: Success from redirect history: {chat_id_to_return}")
                    # else: print("[VCLIENT_WARN] get_chat_id: Redirect but no Location in history.")
            elif response.status_code == 200: # Direct response (less common for new chat ID)
                response_text_direct = response.text.strip()
                if response_text_direct.startswith("cmb"):
                    chat_id_to_return = response_text_direct
                    # print(f"[VCLIENT_DEBUG] get_chat_id: Success from 200 response body: {chat_id_to_return}")
                # else: print(f"[VCLIENT_WARN] get_chat_id: Status 200 but body not a chat_id: '{response_text_direct[:100]}'")
            
            if chat_id_to_return: return chat_id_to_return

            print(f"[VCLIENT_ERROR] get_chat_id: Failed to extract chat_id. Final status: {response.status_code}, Response text: {response.text[:200]}")
            print(f"[VCLIENT_DEBUG] 失效的token对应邮箱: {account_email}")
            if response.status_code >= 400: response.raise_for_status() # Raise for client/server errors if not handled
            return None

        except httpx.HTTPStatusError as e:
            print(f"[VCLIENT_ERROR] get_chat_id: HTTP error for {url_with_force_new}: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except httpx.RequestError as e:
            print(f"[VCLIENT_ERROR] get_chat_id: Request error for {url_with_force_new}: {e}")
            return None
        except Exception as e:
            import traceback
            print(f"[VCLIENT_ERROR] get_chat_id: Unexpected exception for {url_with_force_new}: {e}")
            traceback.print_exc()
            return None

    async def send_message_stream(
        self, auth_token: str, chat_id: str, message_content: str,
        vertical_model_id_str: str, reasoning_on: bool, system_prompt_content: str
    ) -> AsyncGenerator[str, None]:
        
        chat_url = "https://app.verticalstudio.ai/api/chat" 
        payload: Dict[str, Any] = {
            "chatId": chat_id, "cornerType": "text",
            "message": {
                "id": self._generate_message_id(), "createdAt": self._get_iso_timestamp(),
                "role": "user", "content": message_content,
                "parts": [{"type": "text", "text": message_content}]
            },
            "settings": {
                "modelId": vertical_model_id_str, "reasoning": "on" if reasoning_on else "off",
                "systemPromptPreset": system_prompt_content if system_prompt_content and system_prompt_content.strip() else "default",
                "toneOfVoice": "default" 
            }
        }
        headers = {
            "Cookie": f"sb-ppdjlmajmpcqpkdmnzfd-auth-token={auth_token}",
            "Content-Type": "application/json", "Accept": "text/event-stream",
            "User-Agent": USER_AGENT, "Referer": f"https://app.verticalstudio.ai/chat/{chat_id}",
            "Origin": "https://app.verticalstudio.ai"
        }
        
        try:
            async with self.client.stream("POST", chat_url, json=payload, headers=headers, timeout=None) as response:
                if response.status_code not in [200, 202]: 
                    error_text = (await response.aread()).decode('utf-8', errors='replace')
                    print(f"[VCLIENT_ERROR] send_message_stream to {chat_url}: {response.status_code} - {error_text[:200]}")
                    yield f"error: {json.dumps({'message': f'Upstream error {response.status_code}: {error_text[:100]}'})}\n"
                    return

                async for line_bytes in response.aiter_lines():
                    line = line_bytes.strip() 
                    if line: yield line + "\n" 
                        
        except httpx.RequestError as e:
            print(f"[VCLIENT_ERROR] send_message_stream RequestError to {chat_url}: {e}")
            yield f"error: {json.dumps({'message': f'Request error: {str(e)}'})}\n"
        except Exception as e:
            print(f"[VCLIENT_ERROR] send_message_stream Exception to {chat_url}: {e}")
            yield f"error: {json.dumps({'message': f'Internal stream error: {str(e)}'})}\n"
