import json
import os
import random
import time
from typing import Dict, List, Optional, Union

import anthropic
from google import genai
from google.genai import types as genai_types
import pandas as pd
import torch
from ollama import ChatResponse
from ollama import chat as OllamaChat
from openai import AzureOpenAI, OpenAI

from .env_utils import get_env_with_error, get_backbone_llm, get_seed, get_llm_provider


def chat_completion(
    messages: Union[str, List[Dict[str, str]]],
    temperature: float = 0.4,  # Default temperature for balanced creativity and consistency
    model: Optional[str] = None,
    mod: str = 'query',
    attempts: int = 3,
    seed: Optional[int] = None,
    use_openrouter: bool = True,  # Deprecated: use LLM_PROVIDER_NAME env var instead
    response_format: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None
) -> str:
    """
    Unified chat completion function that routes to the provider specified by LLM_PROVIDER_NAME.
    
    Supported providers (set via LLM_PROVIDER_NAME env var):
        - OpenRouter: Routes through OpenRouter unified API (default)
        - Azure: Uses Azure OpenAI API (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
        - OpenAI: Uses official OpenAI API (OPENAI_API_KEY)
        - Claude: Uses Anthropic Claude API (ANTHROPIC_API_KEY)
        - Gemini: Uses Google Gemini API (GEMINI_API_KEY)
    
    Args:
        messages: Either a string prompt or list of message dicts with 'role' and 'content'
        temperature: Sampling temperature (0.0 to 2.0)
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-7-sonnet')
        mod: Message mode - 'query' converts string to user message, 'chat' expects list
        attempts: Number of retry attempts on failure
        seed: Random seed for reproducibility
        use_openrouter: Deprecated - use LLM_PROVIDER_NAME env var instead
        response_format: Optional response format (e.g., {"type": "json_object"} for JSON mode)
        
    Returns:
        Model response content as string
        
    Raises:
        ValueError: If model is not specified
        Exception: If all retry attempts fail
    """
    # Set default model
    if model is None:
        model = get_backbone_llm("gpt-4o")
    
    # Initialize seed
    if seed is None:
        seed = get_seed(default=random.randint(0, 2**32 - 1))
    
    # Format messages
    if mod == 'query' and isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # --- Model-specific overrides (always route to their native provider) ---
    if 'deepseek-r1:671b' in model:
        return _nvidia_deepseek_completion(messages, temperature, attempts, seed)
    if model in ['deepseek-r1:70b', 'llama3.3']:
        return _ollama_completion(messages, model, seed)
    if 'gemini' in model.lower():
        return _gemini_completion(messages, temperature, model)
    if 'claude' in model.lower():
        return _claude_completion(messages, temperature, model, attempts)
    
    # --- Route based on LLM_PROVIDER_NAME (for OpenAI-compatible models) ---
    provider = get_llm_provider()
    
    if provider == "OpenRouter":
        return _openrouter_completion(messages, temperature, model, attempts, seed, response_format)
    elif provider == "Azure":
        return _azure_completion(messages, temperature, model, seed, response_format, reasoning_effort, attempts)
    elif provider == "OpenAI":
        return _openai_completion(messages, temperature, model, attempts, seed, response_format, reasoning_effort)
    elif provider == "Claude":
        return _claude_completion(messages, temperature, model, attempts)
    elif provider == "Gemini":
        return _gemini_completion(messages, temperature, model)
    else:
        print(f"[chat_completion] Unknown provider '{provider}', falling back to OpenRouter", flush=True)
        return _openrouter_completion(messages, temperature, model, attempts, seed, response_format)


def _openrouter_completion(
    messages: List[Dict[str, str]],
    temperature: float,
    model: str,
    attempts: int,
    seed: int,
    response_format: Optional[Dict[str, str]] = None
) -> str:
    """
    Handle completion via OpenRouter unified API.
    
    OpenRouter Documentation: https://openrouter.ai/docs/quickstart
    """
    api_key = get_env_with_error(
        "OPENROUTER_API_KEY",
        required=True,
        description="using OpenRouter API to access LLM models"
    )
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Map common model names to OpenRouter format
    model = _normalize_model_name(model)
    
    for attempt in range(attempts):
        try:
            # print(f"[chat_completion] Using OpenRouter with model: {model}", flush=True)
            
            # Build request parameters
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            # Add response_format if provided and model supports it (OpenAI models)
            if response_format and "openai/" in model:
                request_params["response_format"] = response_format
            
            # Add optional headers for OpenRouter attribution
            extra_headers = {}
            site_url = os.getenv("OPENROUTER_SITE_URL")
            site_name = os.getenv("OPENROUTER_SITE_NAME")
            
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if site_name:
                extra_headers["X-Title"] = site_name
            
            # Some models support seed parameter
            if _model_supports_seed(model):
                request_params["seed"] = seed
            
            # Make request
            if extra_headers:
                completion = client.chat.completions.create(
                    extra_headers=extra_headers,
                    **request_params
                )
            else:
                completion = client.chat.completions.create(**request_params)
            
            return completion.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            wait_time = (2 ** attempt) + 1
            
            print(f"[chat_completion] OpenRouter (model={model}) attempt {attempt + 1}/{attempts} failed: {error_str[:150]}", flush=True)
            
            if attempt < attempts - 1:
                print(f"[chat_completion] Retrying in {wait_time}s...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"[chat_completion] OpenRouter (model={model}) all {attempts} attempts exhausted.", flush=True)
                return f"I cannot help with it - OpenRouter error (model={model}): {error_str[:100]}"
    
    return "I cannot help with it - All retries failed"


def _normalize_model_name(model: str) -> str:
    """
    Normalize model names to OpenRouter format.
    
    OpenRouter uses format: provider/model-name
    Examples: openai/gpt-4o, anthropic/claude-3.5-sonnet, google/gemini-2.0-flash
    """
    # Model mapping for common names
    model_map = {
        # OpenAI models
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-5": "openai/gpt-5",
        "o1-mini": "openai/o1-mini",
        "o1-mini-2024-09-12": "openai/o1-mini-2024-09-12",
        "o3-mini": "openai/o3-mini",
        "o3-mini-0131": "openai/o3-mini",
        "o3-mini-2025-01-31": "openai/o3-mini",
        
        # Anthropic models
        "claude": "anthropic/claude-3.5-sonnet",
        "claude-3-5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-3-7-sonnet": "anthropic/claude-3.7-sonnet",
        
        # Google models
        "gemini": "google/gemini-2.0-flash-exp",
        "gemini-2.0-flash": "google/gemini-2.0-flash",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
        "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite",
        
        # DeepSeek models
        "deepseek-r1": "deepseek/deepseek-r1",
        "deepseek-chat": "deepseek/deepseek-chat",
        
        # NVIDIA DeepSeek models (use legacy handler)
        "deepseek-r1:671b": "deepseek-r1:671b",  # Pass through for legacy handler
    }
    
    # Return mapped name if exists, otherwise assume it's already in correct format
    return model_map.get(model, model)


def _model_supports_seed(model: str) -> bool:
    """Check if model supports seed parameter."""
    seed_supported = ["openai/", "anthropic/"]
    return any(provider in model for provider in seed_supported)


def _nvidia_deepseek_completion(
    messages: List[Dict[str, str]], 
    temperature: float, 
    attempts: int, 
    seed: int
) -> str:
    """Handle NVIDIA DeepSeek R1 completion."""
    try:
        endpoint = get_env_with_error(
            "NVIDIA_DEEPSEEK_ENDPOINT",
            required=True,
            description="connecting to NVIDIA DeepSeek endpoint"
        )
        api_key = get_env_with_error(
            "NVIDIA_DEEPSEEK_API_KEY",
            required=True,
            description="using NVIDIA DeepSeek API"
        )
        
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        
        for attempt in range(attempts):
            try:
                completion = client.chat.completions.create(
                    model="deepseek-ai/deepseek-r1",
                    messages=messages,
                    temperature=temperature,
                    top_p=0.7,
                    max_tokens=4096,
                    seed=seed
                )
                
                content = completion.choices[0].message.content
                
                # Handle reasoning models that use <think> tags
                if '</think>' in content:
                    content = content.split('</think>')[-1]
                
                return content.strip()
                
            except Exception as e:
                print(f"[chat_completion] NVIDIA DeepSeek attempt {attempt + 1}/{attempts} failed: {e}", flush=True)
                if attempt < attempts - 1:
                    time.sleep(4)
                else:
                    return "I cannot help with it - NVIDIA DeepSeek error"
        
        return "I cannot help with it - All NVIDIA DeepSeek attempts failed"
        
    except Exception as e:
        print(f"[chat_completion] NVIDIA DeepSeek initialization error: {e}", flush=True)
        return f"I cannot help with it - NVIDIA DeepSeek initialization error: {str(e)[:100]}"


def _ollama_completion(messages: List[Dict[str, str]], model: str, seed: int) -> str:
    """Handle Ollama model completion."""
    try:
        response: ChatResponse = OllamaChat(
            model=model,
            messages=messages,
            options={"seed": seed}
        )
        
        content = response.message.content
        
        # Handle reasoning models that use <think> tags
        if 'r1' in model and '</think>' in content:
            content = content.split('</think>')[-1].strip()
        
        return content
        
    except Exception as e:
        print(f"[chat_completion] Ollama error (model={model}): {e}", flush=True)
        return f"I cannot help with it - Ollama error (model={model})"


def _gemini_completion(messages: List[Dict[str, str]], temperature: float, model: str) -> str:
    """Handle Google Gemini completion using the new google-genai SDK."""
    wants_json = False
    gemini_model = model if 'gemini' in model.lower() else os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        api_key = get_env_with_error(
            "GEMINI_API_KEY",
            required=True,
            description="using Google Gemini API"
        )
        client = genai.Client(api_key=api_key)

        # Separate system instructions from conversation history
        system_parts = []
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=content)]))
            elif role == "assistant":
                contents.append(genai_types.Content(role="model", parts=[genai_types.Part(text=content)]))

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        last_content = messages[-1]["content"] if messages else ""
        wants_json = any(kw in last_content.lower() for kw in [
            "json format", "output in json", "'reasoning'", '"reasoning"',
            "confidence_level", "{'reasoning'"
        ])

        config_kwargs = {"temperature": temperature, "candidate_count": 1}
        if wants_json:
            config_kwargs["response_mime_type"] = "application/json"
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = genai_types.GenerateContentConfig(**config_kwargs)

        time.sleep(1)
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=config,
        )
        return response.text

    except Exception as e:
        if wants_json and "response_mime_type" in str(e):
            try:
                config_kwargs.pop("response_mime_type", None)
                config = genai_types.GenerateContentConfig(**config_kwargs)
                time.sleep(1)
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as retry_e:
                print(f"[chat_completion] Gemini retry without JSON mode also failed (model={gemini_model}): {retry_e}", flush=True)
                return f"I cannot help with it - Gemini error (model={gemini_model})"
        print(f"[chat_completion] Gemini error (model={gemini_model}): {e}", flush=True)
        return f"I cannot help with it - Gemini error (model={gemini_model})"


def _claude_completion(
    messages: List[Dict[str, str]],
    temperature: float,
    model: str,
    attempts: int,
) -> str:
    """Handle Anthropic Claude API completion."""
    try:
        api_key = get_env_with_error(
            "ANTHROPIC_API_KEY",
            required=True,
            description="using Anthropic Claude API"
        )
        
        _request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
        client = anthropic.Anthropic(api_key=api_key, timeout=_request_timeout)
        
        # Separate system message from conversation messages
        system_msg = None
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                claude_messages.append(msg)
        
        # If the model name is not a Claude model, use ANTHROPIC_MODEL from env
        claude_model = model
        if 'claude' not in model.lower():
            claude_model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
        
        for attempt in range(attempts):
            try:
                request_params = {
                    "model": claude_model,
                    "max_tokens": 4096,
                    "messages": claude_messages,
                    "temperature": temperature,
                }
                if system_msg:
                    request_params["system"] = system_msg
                
                response = client.messages.create(**request_params)
                return response.content[0].text
                
            except Exception as e:
                wait_time = (2 ** attempt) + 1
                print(f"[chat_completion] Claude (model={claude_model}) attempt {attempt + 1}/{attempts} failed: {str(e)[:150]}", flush=True)
                if attempt < attempts - 1:
                    print(f"[chat_completion] Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                else:
                    return f"I cannot help with it - Claude error: {str(e)[:100]}"
        
        return "I cannot help with it - All Claude attempts failed"
        
    except Exception as e:
        print(f"[chat_completion] Claude initialization error (model={model}): {e}", flush=True)
        return f"I cannot help with it - Claude error (model={model}): {str(e)[:100]}"


def _openai_completion(
    messages: List[Dict[str, str]],
    temperature: float,
    model: str,
    attempts: int,
    seed: int,
    response_format: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None
) -> str:
    """Handle direct OpenAI API completion (not Azure, not OpenRouter)."""
    try:
        api_key = get_env_with_error(
            "OPENAI_API_KEY",
            required=True,
            description="using OpenAI API"
        )
        
        _request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
        client = OpenAI(api_key=api_key, timeout=_request_timeout)
        
        # Build request params
        request_params = {
            "model": model,
            "messages": messages,
        }
        
        # Models that use max_completion_tokens instead of temperature/seed:
        # o-series (o1, o3, o4) and gpt-5 series
        _no_temp_models = ('o1' in model or 'o3' in model or 'o4' in model
                           or 'gpt-5' in model)
        
        if _no_temp_models:
            default_max = 16384 if 'gpt-5' in model else 4096
            if 'gpt-5' in model and model not in ('gpt-5', 'gpt-5-mini') and reasoning_effort is not None:
                request_params["reasoning_effort"] = reasoning_effort
                _effort_token_map = {"none": 4096, "low": 4096, "medium": 8192, "high": 16384, "xhigh": 16384}
                default_max = _effort_token_map.get(reasoning_effort, default_max)
            request_params["max_completion_tokens"] = int(os.getenv("MAX_COMPLETION_TOKENS", str(default_max)))
        else:
            request_params["temperature"] = temperature
            request_params["seed"] = seed
        
        # Add response_format if provided and supported
        if response_format and not _no_temp_models:
            request_params["response_format"] = response_format
        
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(**request_params)
                return response.choices[0].message.content
            except Exception as e:
                wait_time = (2 ** attempt) + 1
                print(f"[chat_completion] OpenAI (model={model}) attempt {attempt + 1}/{attempts} failed: {str(e)[:150]}", flush=True)
                if attempt < attempts - 1:
                    print(f"[chat_completion] Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                else:
                    return f"I cannot help with it - OpenAI error (model={model}): {str(e)[:100]}"
        
        return "I cannot help with it - All OpenAI attempts failed"
        
    except Exception as e:
        print(f"[chat_completion] OpenAI initialization error (model={model}): {e}", flush=True)
        return f"I cannot help with it - OpenAI error (model={model}): {str(e)[:100]}"


def _azure_completion(
    messages: List[Dict[str, str]], 
    temperature: float, 
    model: str, 
    seed: int,
    response_format: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None,
    attempts: int = 3,
) -> str:
    """Handle Azure OpenAI completion."""
    try:
        # Determine API version
        if 'o1-mini' in model:
            api_version = os.getenv("O1_MINI_API_VERSION", "2024-12-01-preview")
        elif 'o3-mini' in model:
            api_version = os.getenv("O3_MINI_API_VERSION", "2024-12-01-preview")
        elif 'o4-mini' in model:
            api_version = os.getenv("O4_MINI_API_VERSION", "2024-12-01-preview")
        else:
            api_version = get_env_with_error("AZURE_API_VERSION", default="2024-10-21")
        
        api_key = get_env_with_error(
            "AZURE_OPENAI_API_KEY",
            required=True,
            description="using Azure OpenAI API"
        )
        endpoint = get_env_with_error(
            "AZURE_OPENAI_ENDPOINT",
            required=True,
            description="connecting to Azure OpenAI endpoint"
        )
        
        # Ensure endpoint uses https
        if endpoint.startswith("http://"):
            endpoint = endpoint.replace("http://", "https://")
        
        _request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            timeout=_request_timeout,
            max_retries=0,
        )
        
        # Build request params
        request_params = {
            "model": model,
            "messages": messages,
        }
        
        # Models that use max_completion_tokens instead of temperature/seed:
        # o-series (o1, o3, o4) and gpt-5 series
        _no_temp_models = ('o1' in model or 'o3' in model or 'o4' in model
                           or 'gpt-5' in model)
        
        if _no_temp_models:
            default_max = 16384 if 'gpt-5' in model else 4096
            # Scale token budget by reasoning effort for GPT-5.2+
            if 'gpt-5' in model and model not in ('gpt-5', 'gpt-5-mini') and reasoning_effort is not None:
                request_params["reasoning_effort"] = reasoning_effort
                _effort_token_map = {"none": 4096, "low": 4096, "medium": 8192, "high": 16384, "xhigh": 16384}
                default_max = _effort_token_map.get(reasoning_effort, default_max)
            request_params["max_completion_tokens"] = int(os.getenv("MAX_COMPLETION_TOKENS", str(default_max)))
        else:
            request_params["temperature"] = temperature
            request_params["seed"] = seed
        
        # Add response_format if provided and supported
        if response_format and not _no_temp_models:
            request_params["response_format"] = response_format
        
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(**request_params)
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                wait_time = min((2 ** attempt) + 1, 30)
                print(f"[chat_completion] Azure error (model={model}) attempt {attempt+1}/{attempts}: {error_str[:150]}", flush=True)
                if attempt < attempts - 1:
                    print(f"[chat_completion] Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                else:
                    return f"I cannot help with it - Azure error (model={model}): {error_str[:100]}"
        return f"I cannot help with it - Azure error (model={model}): all {attempts} attempts failed"

    except Exception as e:
        print(f"[chat_completion] Azure error (model={model}): {e}", flush=True)
        return f"I cannot help with it - Azure error (model={model}): {str(e)[:100]}"


def web_search_completion(
    query: str,
    model: Optional[str] = None,
    search_context_size: str = "medium",
) -> str:
    """
    Web search-augmented completion using the OpenAI Responses API.
    
    Uses `web_search_preview` tool so the model can search the web before answering.
    Only supported with Azure and OpenAI providers; falls back to chat_completion otherwise.
    
    Args:
        query: The user query / prompt
        model: Model identifier (e.g. 'gpt-5')
        search_context_size: Amount of search context — 'low', 'medium', or 'high'
    """
    if model is None:
        model = get_backbone_llm("gpt-4o")

    provider = get_llm_provider()

    try:
        if provider == "Azure":
            return _azure_web_search(query, model, search_context_size)
        elif provider == "OpenAI":
            return _openai_web_search(query, model, search_context_size)
        else:
            print(
                f"[web_search_completion] Web search not supported for provider '{provider}'. "
                f"Falling back to chat_completion.",
                flush=True
            )
            return chat_completion(query, model=model)
    except Exception as e:
        print(f"[web_search_completion] Failed: {e}. Falling back to chat_completion.", flush=True)
        return chat_completion(query, model=model)


def _responses_api_call(client, model: str, query: str, search_context_size: str) -> str:
    """Shared Responses API call for web search."""
    response = client.responses.create(
        model=model,
        tools=[{
            "type": "web_search_preview",
            "search_context_size": search_context_size,
        }],
        input=query,
    )
    # Extract text from the Responses API output
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for part in item.content:
                if getattr(part, "type", None) == "output_text":
                    return part.text
    return str(response.output)


def _azure_web_search(query: str, model: str, search_context_size: str) -> str:
    """Azure OpenAI web search via Responses API."""
    api_key = get_env_with_error(
        "AZURE_OPENAI_API_KEY", required=True,
        description="Azure web search"
    )
    endpoint = get_env_with_error(
        "AZURE_OPENAI_ENDPOINT", required=True,
        description="Azure web search"
    )
    if endpoint.startswith("http://"):
        endpoint = endpoint.replace("http://", "https://")
    base = endpoint.rstrip("/")

    client = OpenAI(
        api_key=api_key,
        base_url=f"{base}/openai/v1/",
    )
    return _responses_api_call(client, model, query, search_context_size)


def _openai_web_search(query: str, model: str, search_context_size: str) -> str:
    """OpenAI direct web search via Responses API."""
    api_key = get_env_with_error(
        "OPENAI_API_KEY", required=True,
        description="OpenAI web search"
    )
    client = OpenAI(api_key=api_key)
    return _responses_api_call(client, model, query, search_context_size)


def form_ppi_embed_dict(celltype_ppi_embed, celltype_dict, celltype_protein_dict):
    # each node(gene) has a vector representation dim: (128,)
    ppi_embed_dict = {}
    for celltype, index in celltype_dict.items():
        cell_embed_dict = {}
        cell_embed = celltype_ppi_embed[index]
        for i, gene in enumerate(celltype_protein_dict[celltype]):
            gene_embed = cell_embed[i, :]
            cell_embed_dict[gene] = gene_embed
            # print(f"[pinnacle]: {celltype} - {gene} - {gene_embed.shape}")
        celltype = celltype.replace(" ", "_")
        ppi_embed_dict[celltype] = cell_embed_dict
        # print(f"[pinnacle]: {celltype} - {len(cell_embed_dict)}")
    return ppi_embed_dict

def load_embed_only(embed_path: str, labels_path: str):
    embed = torch.load(embed_path)
    with open(labels_path, "r") as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)
    celltypes = [c for c in labels_dict["Cell Type"] if c.startswith("CCI")]
    celltype_dict = {ct.split("CCI_")[1]: i for i, ct in enumerate(celltypes)}
    assert len(celltype_dict) > 0
    
    protein_names = []
    protein_celltypes = []
    for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"): continue
        protein_names.append(p)
        protein_celltypes.append(c)

    proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
    celltype_protein_dict = proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list}).to_dict()["target"]
    assert len(celltype_protein_dict) > 0
    return embed, celltype_dict, celltype_protein_dict