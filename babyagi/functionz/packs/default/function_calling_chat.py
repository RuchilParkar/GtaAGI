from functionz.core.framework import func
import json
import litellm

# Assuming `func` is the registry from your framework
# and `execute_function_wrapper` is already registered in the database.

@func.register_function(
    metadata={
        "description": "A chat application that interacts with LiteLLM and executes selected functions from the database."
    },
    imports=["litellm", "json", "os"],
    dependencies=["get_function_wrapper", "execute_function_wrapper"]
)
def chat_with_functions(chat_history, available_function_names) -> str:
    def map_python_type_to_json(python_type: str) -> dict:
        """
        Maps Python type annotations to JSON Schema types.

        Args:
            python_type (str): The Python type as a string.

        Returns:
            dict: The corresponding JSON Schema type with additional details if necessary.
        """
        type_mapping = {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list": {"type": "array", "items": {"type": "string"}},  # Assuming list of strings
            "dict": {"type": "object"},
            "Any": {"type": "string"}  # Default to string for unsupported types
        }
        return type_mapping.get(python_type, {"type": "string"})

    # Enable verbose logging for LiteLLM
    litellm.set_verbose = True

    # Initialize chat context with system message
    chat_context = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # Validate and append chat history
    if not isinstance(chat_history, list):
        raise ValueError("chat_history must be a list of messages.")

    for message in chat_history:
        if not isinstance(message, dict):
            raise ValueError("Each message in chat_history must be a dictionary.")
        role = message.get('role')
        content = message.get('message')
        if role not in ['user', 'assistant', 'system']:
            raise ValueError("Message role must be 'user', 'assistant', or 'system'.")
        if not isinstance(content, str):
            raise ValueError("Message content must be a string.")
        chat_context.append({"role": role, "content": content})

    # Handle available_function_names input
    if isinstance(available_function_names, str):
        # Split the string by commas and strip whitespace
        available_function_names = [name.strip() for name in available_function_names.split(',') if name.strip()]
    elif isinstance(available_function_names, list):
        # Ensure all elements are strings and strip whitespace
        available_function_names = [name.strip() for name in available_function_names if isinstance(name, str) and name.strip()]
    else:
        raise ValueError("available_function_names must be a string or a list of strings.")

    if not available_function_names:
        raise ValueError("No valid function names provided in available_function_names.")

    # Fetch available functions from the database
    tools = []
    for func_name in available_function_names:
        # Retrieve function details using the get_function_wrapper
        function_data = get_function_wrapper(func_name)
        if function_data:
            # Construct the tool definition for LiteLLM
            tool = {
                "type": "function",
                "function": {
                    "name": function_data['name'],
                    "description": (function_data.get('metadata') or {}).get('description', f"Function {function_data['name']}"),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
            }

            # Map input_parameters to the tool's parameters
            for param in function_data.get('input_parameters', []):
                # Convert Python types to JSON Schema types
                json_schema = map_python_type_to_json(param['type'])
                tool['function']['parameters']['properties'][param['name']] = {
                    **json_schema,
                    "description": param.get('description', '')
                }
                if param.get('required', False):
                    tool['function']['parameters']['required'].append(param['name'])

            tools.append(tool)
        else:
            # Handle the case where the function is not found
            raise ValueError(f"Function '{func_name}' not found in the database.")


    import os
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        try:
            import litellm.llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini as gemini_module
            import litellm.llms.vertex_ai_and_google_ai_studio.gemini.transformation as trans_module

            if not getattr(gemini_module, "_thought_sig_patched", False):
                orig_transform_parts = gemini_module.VertexGeminiConfig._transform_parts

                def _patched_transform_parts(self, parts, index, is_function_call=False):
                    function, tools_list = orig_transform_parts(self, parts, index, is_function_call)
                    if tools_list:
                        for i, part in enumerate(parts):
                            if "functionCall" in part and i < len(tools_list):
                                ts = part.get("thoughtSignature") or part.get("thought_signature")
                                if ts:
                                    tools_list[i]["thoughtSignature"] = ts
                    return function, tools_list

                gemini_module.VertexGeminiConfig._transform_parts = _patched_transform_parts

                orig_convert = trans_module.convert_to_gemini_tool_call_invoke

                def _patched_convert(message):
                    parts = orig_convert(message)
                    t_calls = message.get("tool_calls", [])
                    for i, tc in enumerate(t_calls):
                        ts = tc.get("thoughtSignature") if isinstance(tc, dict) else getattr(tc, "thoughtSignature", None)
                        if not ts:
                            ts = tc.get("thought_signature") if isinstance(tc, dict) else getattr(tc, "thought_signature", None)
                        if ts and i < len(parts):
                            parts[i]["thoughtSignature"] = ts
                    return parts

                trans_module.convert_to_gemini_tool_call_invoke = _patched_convert
                gemini_module._thought_sig_patched = True
        except Exception:
            pass

    def _call_completion(messages, tools=None):
        if gemini_key:
            preferred = os.getenv("GEMINI_MODEL")
            candidates = [preferred] if preferred else ["gemini/gemini-3.1-flash-lite", "gemini/gemini-3.5-flash", "gemini/gemini-2.5-flash"]
            last_err = None
            for model in candidates:
                if not model:
                    continue
                try:
                    kwargs = {"model": model, "messages": messages}
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = "auto"
                    return litellm.completion(**kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    if "not found" in err_str or "404" in err_str or "not supported" in err_str:
                        last_err = e
                        continue
                    raise
            if last_err:
                raise last_err
        elif openai_key:
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return litellm.completion(**kwargs)
        else:
            raise ValueError("No API key configured. Please set GEMINI_API_KEY or OPENAI_API_KEY in .env")

    # Multi-turn tool execution loop
    max_turns = 5
    for _ in range(max_turns):
        response = _call_completion(chat_context, tools=tools)
        response_message = response['choices'][0]['message']
        tool_calls = response_message.get('tool_calls', [])

        if not tool_calls:
            return response_message.get('content', '')

        # Append the assistant's message to the chat context
        chat_context.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_args = json.loads(tool_call['function']['arguments'])
            tool_call_id = tool_call['id']

            # Execute the function using execute_function_wrapper
            try:
                function_response = execute_function_wrapper(function_name, **function_args)
            except Exception as e:
                function_response = f"Error executing function '{function_name}': {str(e)}"

            # Ensure function_response is a string
            if not isinstance(function_response, str):
                function_response = json.dumps(function_response)

            # Append the function response to the chat context
            chat_context.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })

    return response_message.get('content', '')
