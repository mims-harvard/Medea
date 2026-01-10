import json
import os
import random
import time
from typing import Dict, List, Optional, Union

import anthropic
import google.generativeai as genai
import pandas as pd
import torch
from ollama import ChatResponse
from ollama import chat as OllamaChat
from openai import AzureOpenAI, OpenAI

from .env_utils import get_env_with_error, get_backbone_llm, get_seed


def chat_completion(
    messages: Union[str, List[Dict[str, str]]],
    temperature: float = 0.4,  # Default temperature for balanced creativity and consistency
    model: Optional[str] = None,
    mod: str = 'query',
    attemps: int = 3,
    seed: Optional[int] = None,
    use_openrouter: bool = True,
    response_format: Optional[Dict[str, str]] = None
) -> str:
    """
    Unified chat completion function using OpenRouter as the primary API gateway.
    
    OpenRouter provides a single endpoint to access 100+ AI models from different providers,
    handling fallbacks and rate limits automatically. See: https://openrouter.ai/docs/quickstart
    
    Args:
        messages: Either a string prompt or list of message dicts with 'role' and 'content'
        temperature: Sampling temperature (0.0 to 2.0)
        model: Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')
        mod: Message mode - 'query' converts string to user message, 'chat' expects list
        attemps: Number of retry attempts on failure
        seed: Random seed for reproducibility
        use_openrouter: If True, routes through OpenRouter; if False, uses direct APIs
        response_format: Optional response format (e.g., {"type": "json_object"} for JSON mode)
        
    Returns:
        Model response content as string
        
    Raises:
        ValueError: If model is not specified
        Exception: If all retry attempts fail
    """
    # Set default model
    if model is None:
        model = get_backbone_llm("openai/gpt-4o")
    
    # Initialize seed
    if seed is None:
        seed = get_seed(default=random.randint(0, 2**32 - 1))
    
    # Format messages
    if mod == 'query' and isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # Route to appropriate handler
    # Force legacy handler for certain models that need direct API access
    if 'deepseek-r1:671b' in model or not use_openrouter:
        return _legacy_completion(messages, temperature, model, attemps, seed, response_format)
    else:
        return _openrouter_completion(messages, temperature, model, attemps, seed, response_format)


def _openrouter_completion(
    messages: List[Dict[str, str]],
    temperature: float,
    model: str,
    attemps: int,
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
    
    for attempt in range(attemps):
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
            
            print(f"[chat_completion] Attempt {attempt + 1}/{attemps} failed: {error_str[:150]}", flush=True)
            
            if attempt < attemps - 1:
                print(f"[chat_completion] Retrying in {wait_time}s...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"[chat_completion] All {attemps} attempts exhausted.", flush=True)
                return f"I cannot help with it - Error: {error_str[:100]}"
    
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


def _legacy_completion(
    messages: List[Dict[str, str]],
    temperature: float,
    model: str,
    attemps: int,
    seed: int,
    response_format: Optional[Dict[str, str]] = None
) -> str:
    """
    Legacy completion handler for direct API calls (non-OpenRouter).
    Kept for backward compatibility and special cases.
    """
    # Handle NVIDIA DeepSeek models
    if 'deepseek-r1:671b' in model:
        return _nvidia_deepseek_completion(messages, temperature, attemps, seed)
    
    # Handle Ollama models
    if model in ['deepseek-r1:70b', 'llama3.3']:
        return _ollama_completion(messages, model, seed)
    
    # Handle Gemini models
    if 'gemini' in model:
        return _gemini_completion(messages, temperature, model)
    
    # Handle Azure OpenAI models
    if model in ['o3-mini', 'o3-mini-0131', 'o1-mini-2025-03-01', 'o4-mini-0416']:
        return _azure_completion(messages, temperature, model, seed, response_format)
    
    # Default: treat as Azure
    return _azure_completion(messages, temperature, model, seed, response_format)


def _nvidia_deepseek_completion(
    messages: List[Dict[str, str]], 
    temperature: float, 
    attemps: int, 
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
        
        for attempt in range(attemps):
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
                print(f"[chat_completion] NVIDIA DeepSeek attempt {attempt + 1}/{attemps} failed: {e}", flush=True)
                if attempt < attemps - 1:
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
        print(f"[chat_completion] Ollama error: {e}", flush=True)
        return "I cannot help with it - Ollama error"


def _gemini_completion(messages: List[Dict[str, str]], temperature: float, model: str) -> str:
    """Handle Google Gemini completion."""
    try:
        api_key = get_env_with_error(
            "GEMINI_API_KEY",
            required=True,
            description="using Google Gemini API"
        )
        genai.configure(api_key=api_key)
        
        # Convert to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                gemini_messages.append({"role": "system", "content": content})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})
        
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=temperature,
        )
        
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        model_instance = genai.GenerativeModel(gemini_model, generation_config=generation_config)
        chat = model_instance.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        
        time.sleep(1)  # Rate limiting
        response = chat.send_message(messages[-1]["content"])
        return response.text
        
    except Exception as e:
        print(f"[chat_completion] Gemini error: {e}", flush=True)
        return "I cannot help with it - Gemini error"


def _azure_completion(
    messages: List[Dict[str, str]], 
    temperature: float, 
    model: str, 
    seed: int,
    response_format: Optional[Dict[str, str]] = None
) -> str:
    """Handle Azure OpenAI completion."""
    try:
        # Determine API version
        if 'o1-mini' in model:
            api_version = os.getenv("O1_MINI_API_VERSION")
        elif 'o3-mini' in model:
            api_version = os.getenv("O3_MINI_API_VERSION")
        elif 'o4-mini' in model:
            api_version = os.getenv("O4_MINI_API_VERSION")
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
        
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        # Build request params
        request_params = {
            "model": model,
            "messages": messages,
        }
        
        # Some models don't support temperature or seed
        if 'o1' not in model and 'o3' not in model and 'o4' not in model:
            request_params["temperature"] = temperature
            request_params["seed"] = seed
        
        # Add response_format if provided and supported
        if response_format and 'o1' not in model and 'o3' not in model and 'o4' not in model:
            request_params["response_format"] = response_format
        
        response = client.chat.completions.create(**request_params)
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"[chat_completion] Azure error: {e}", flush=True)
        return f"I cannot help with it - Azure error: {str(e)[:100]}"


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