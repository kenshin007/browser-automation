import pdb
import logging

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

import gradio as gr

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot


# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()

def save_test_case(test_case_name, task_description, verification_condition):
    """Save a test case to a JSON file."""
    import json
    import os
    import datetime
    
    # Create test cases directory if it doesn't exist
    test_cases_dir = "test_cases"
    os.makedirs(test_cases_dir, exist_ok=True)
    
    # Generate a filename based on the test case name
    filename = test_case_name.strip()
    if not filename:
        filename = "Untitled Test"
    
    # Add timestamp to ensure uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.json"
    
    # Clean filename of invalid characters
    filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    filename = os.path.join(test_cases_dir, filename)
    
    # Create test case data
    test_case = {
        "name": test_case_name,
        "task_description": task_description,
        "verification_condition": verification_condition,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    # Save to file
    try:
        with open(filename, "w") as f:
            json.dump(test_case, f, indent=2)
        return f"Test case saved to {filename}", gr.update()  # Don't try to update the file component
    except Exception as e:
        return f"Error saving test case: {str(e)}", gr.update()

def load_test_case(test_case_file):
    """Load a test case from a JSON file."""
    import json
    
    try:
        if not test_case_file:
            return "No file selected", gr.update(), gr.update(), gr.update()
        
        with open(test_case_file.name, "r") as f:
            test_case = json.load(f)
        
        return (
            f"Test case '{test_case['name']}' loaded successfully",
            gr.update(value=test_case.get("name", "")),
            gr.update(value=test_case.get("task_description", "")),
            gr.update(value=test_case.get("verification_condition", ""))
        )
    except Exception as e:
        return f"Error loading test case: {str(e)}", gr.update(), gr.update(), gr.update()

def list_test_cases():
    """List all available test cases."""
    import os
    import glob
    import json
    
    test_cases_dir = "test_cases"
    if not os.path.exists(test_cases_dir):
        return []
    
    test_case_files = glob.glob(os.path.join(test_cases_dir, "*.json"))
    test_cases = []
    
    for file_path in test_case_files:
        try:
            with open(file_path, "r") as f:
                test_case = json.load(f)
                test_cases.append((file_path, f"{test_case.get('name', 'Unnamed')} - {test_case.get('created_at', '')}"))
        except:
            # Skip files that can't be parsed
            continue
    
    return test_cases

def update_llm_num_ctx_visibility(provider):
    """Show/hide the context length slider based on the provider."""
    return gr.update(visible=provider == "ollama")

def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text
        
    import re
    
    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)
    
    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)
        
    return result

async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser, _global_agent

    try:
        # Request stop
        _global_agent.stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            message,                                        # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )
        
async def stop_research_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (                                   # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp
):
    global _global_agent_state
    _global_agent_state.clear_stop()  # Clear any previous stop requests

    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',                                         # final_result
            errors,                                     # errors
            '',                                         # model_actions
            '',                                         # model_thoughts
            None,                                       # latest_video
            None,                                       # history_file
            None,                                       # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent
        
        # Clear any previous stop request
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None
            
        if _global_browser is None:

            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    cdp_url=cdp_url,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent

        # Clear any previous stop request
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp
        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)

            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        #if chrome_cdp not empty string nor None
        if ((_global_browser is None) or (cdp_url and cdp_url != "" and cdp_url != None)) :
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if (_global_browser_context is None  or (chrome_cdp and cdp_url != "" and cdp_url != None)):
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )


        # Create and run agent
        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)        

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_with_stream(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_num_ctx,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method,
    chrome_cdp
):
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            chrome_cdp=chrome_cdp
        )
        # Add HTML content at the start of the result array
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        yield [html_content] + list(result)

    else:
        try:
            _global_agent_state.clear_stop()
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_num_ctx=llm_num_ctx,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method,
                    chrome_cdp=chrome_cdp
                )
            )

            # Initialize values for streaming
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            latest_videos = trace = history_file = None


            # Periodically update the stream while the agent task is running
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception as e:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent_state and _global_agent_state.is_stop_requested():
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),  # stop_button
                        gr.update(interactive=False),  # run_button
                    ]
                    break
                else:
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stop", interactive=True),  # Re-enable stop button
                        gr.update(interactive=True)  # Re-enable run button
                    ]
                await asyncio.sleep(0.05)

            # Once the agent task completes, get the results
            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                latest_videos = trace = history_file = None

            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                html_content,
                final_result,
                errors,
                model_actions,
                model_thoughts,
                latest_videos,
                trace,
                history_file,
                stop_button,
                run_button
            ]

        except Exception as e:
            import traceback
            yield [
                f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True)    # Re-enable run button
            ]

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}

async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None
        
async def run_deep_search(research_task, max_search_iteration_input, max_query_per_iter_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision, use_own_browser, headless, chrome_cdp):
    from src.utils.deep_research import deep_research
    global _global_agent_state

    # Clear any previous stop request
    _global_agent_state.clear_stop()
    
    llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
    markdown_content, file_path = await deep_research(research_task, llm, _global_agent_state,
                                                        max_search_iterations=max_search_iteration_input,
                                                        max_query_num=max_query_per_iter_input,
                                                        use_vision=use_vision,
                                                        headless=headless,
                                                        use_own_browser=use_own_browser,
                                                        chrome_cdp=chrome_cdp
                                                        )
    
    return markdown_content, file_path, gr.update(value="Stop", interactive=True),  gr.update(interactive=True) 
    

async def run_verification_test(
    test_case_name, task_description, verification_condition,
    llm_provider, llm_model_name, llm_temperature, llm_num_ctx,
    max_steps, max_actions_per_step, use_vision, tool_calling_method,
    use_own_browser, keep_browser_open, headless, window_w, window_h,
    save_recording_path, save_trace_path
):
    """Run the agent with the specified configurations for verification testing."""
    global _global_browser, _global_browser_context, _global_agent, _global_agent_state
    
    # Clear any previous stop requests
    _global_agent_state.clear_stop()
    
    if not task_description.strip() or not verification_condition.strip():
        error_html = """
        <div class="test-result-container">
            <div class="verification-failed">ERROR</div>
            <div class="test-details failed">
                Task Description and Verification Condition are required.
            </div>
        </div>
        """
        return "Error: Task Description and Verification Condition are required.", error_html

    # Combine task description and verification condition
    full_task = f"{task_description.strip()} and {verification_condition.strip()}"
    
    # Create LLM instance
    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        num_ctx=llm_num_ctx if llm_provider == "ollama" else None
    )
    
    # Browser setup
    extra_chromium_args = []
    chrome_path = None
    
    if use_own_browser:
        chrome_path = os.getenv("CHROME_PATH", None)
        if chrome_path:
            extra_chromium_args += [f"--user-data-dir={os.getenv('CHROME_USER_DATA', '')}"]
    
    controller = CustomController()
    
    # Initialize browser if needed
    if _global_browser is None:
        _global_browser = CustomBrowser(
            config=BrowserConfig(
                headless=headless,
                disable_security=True,
                chrome_instance_path=chrome_path,
                extra_chromium_args=extra_chromium_args,
            )
        )
    
    if _global_browser_context is None:
        _global_browser_context = await _global_browser.new_context(
            config=BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path if save_recording_path else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(
                    width=window_w, height=window_h
                ),
            )
        )
    
    # Create and run agent
    _global_agent = CustomAgent(
        task=full_task,
        add_infos="",
        use_vision=use_vision,
        llm=llm,
        browser=_global_browser,
        browser_context=_global_browser_context,
        controller=controller,
        system_prompt_class=CustomSystemPrompt,
        agent_prompt_class=CustomAgentMessagePrompt,
        max_actions_per_step=max_actions_per_step,
        tool_calling_method=tool_calling_method
    )
    
    try:
        history = await _global_agent.run(max_steps=max_steps)
        
        # Parse the final result
        final_result = history.final_result()
        
        # Format the test case name
        formatted_test_name = test_case_name if test_case_name.strip() else "Untitled Test"
        
        # Initialize verification variables
        verification_passed = False
        verification_failed = False
        verification_text = ""
        failure_reason = ""
        
        # Check history for done actions
        if hasattr(history, 'history') and history.history:
            # Iterate through history items in reverse order to find the most recent done action
            for history_item in reversed(history.history):
                if hasattr(history_item, 'model_output') and history_item.model_output and hasattr(history_item.model_output, 'action'):
                    for action in history_item.model_output.action:
                        # Check if this is a done action by looking for a 'done' key in the action dict
                        action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
                        if 'done' in action_dict and action_dict['done']:
                            # Extract the text from the done action
                            done_text = action_dict['done'].get('text', '')
                            if "Verification: Passed" in done_text:
                                verification_passed = True
                                verification_text = done_text
                                break
                            elif "Verification: Failed" in done_text:
                                verification_failed = True
                                verification_text = done_text
                                # Try to extract failure reason
                                if ":" in done_text.split("Verification: Failed")[1]:
                                    failure_reason = done_text.split("Verification: Failed")[1].split(":")[1].strip()
                                break
                    if verification_passed or verification_failed:
                        break
        
        # If we couldn't find a done action with verification text, check the model_actions_log
        if not verification_passed and not verification_failed:
            model_actions_log = history.model_actions() if hasattr(history, 'model_actions') else ""
            
            if model_actions_log and isinstance(model_actions_log, str):
                import re
                
                # Look for the done action with verification text
                done_matches = re.findall(r'(?:{"done":\s*{"text":\s*"([^"]+)"}}|Action \d+/\d+: {"done":{"text":"([^"]+)"}})', model_actions_log)
                
                if done_matches:
                    for match in done_matches:
                        # If it's a tuple (from the regex groups), get the non-empty value
                        if isinstance(match, tuple):
                            done_text = next((m for m in match if m), "")
                        else:
                            done_text = match
                        
                        if "Verification: Passed" in done_text:
                            verification_passed = True
                            verification_text = done_text
                            break
                        elif "Verification: Failed" in done_text:
                            verification_failed = True
                            verification_text = done_text
                            # Try to extract failure reason
                            if ":" in done_text.split("Verification: Failed")[1]:
                                failure_reason = done_text.split("Verification: Failed")[1].split(":")[1].strip()
                            break
        
        # If we still couldn't find verification text, check the final result
        if not verification_passed and not verification_failed and final_result:
            if "Verification: Passed" in final_result:
                verification_passed = True
                verification_text = final_result
            elif "Verification: Failed" in final_result:
                verification_failed = True
                verification_text = final_result
        
        # Generate the appropriate HTML based on verification result
        if verification_passed:
            result_html = f"""
            <div class="test-result-container">
                <div class="test-result-header">Test: {formatted_test_name}</div>
                <div class="verification-passed">✅ PASSED</div>
                <div class="test-details passed">
                    <strong>Verification Status:</strong> {verification_text}
                    <br><br><strong>Agent Output:</strong> {final_result}
                </div>
            </div>
            """
            status = "Passed"
        elif verification_failed:
            result_html = f"""
            <div class="test-result-container">
                <div class="test-result-header">Test: {formatted_test_name}</div>
                <div class="verification-failed">❌ FAILED</div>
                <div class="test-details failed">
                    <strong>Verification Status:</strong> {verification_text}
                    <br><br><strong>Reason:</strong> {failure_reason if failure_reason else "Test conditions were not met."}
                    <br><br><strong>Agent Output:</strong> {final_result}
                </div>
            </div>
            """
            status = "Failed"
        else:
            # Truly unclear result
            result_html = f"""
            <div class="test-result-container">
                <div class="test-result-header">Test: {formatted_test_name}</div>
                <div class="verification-unclear">⚠️ UNCLEAR</div>
                <div class="test-details unclear">
                    <strong>The agent did not clearly indicate verification status.</strong>
                    <br><br><strong>Agent Output:</strong> {final_result if final_result else "No result returned from agent."}
                </div>
            </div>
            """
            if final_result:
                final_result += "\nNote: Agent did not clearly indicate verification status."
            else:
                final_result = "No result returned from agent."
            status = "Unclear"
        
        # Log the test result
        log_test_result(test_case_name, task_description, verification_condition, status)
        
        return final_result, result_html
    except Exception as e:
        error_message = f"Error running verification test: {str(e)}"
        logger.error(error_message)
        error_html = f"""
        <div class="test-result-container">
            <div class="verification-failed">❌ ERROR</div>
            <div class="test-details failed">
                {error_message}
            </div>
        </div>
        """
        return error_message, error_html
    finally:
        # Clean up resources if not keeping browser open
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None
            if _global_browser:
                await _global_browser.close()
                _global_browser = None
            _global_agent = None

def log_test_result(test_name, task, verification, result):
    """Log test results to task-logs.md file."""
    import datetime
    
    log_file = "task-logs.md"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create log entry
    log_entry = f"""
## {test_name}

GOAL: {task} and {verification}
IMPLEMENTATION: Ran verification test using the agent with verification capabilities.
RESULT: {result}
COMPLETED: {now}

---
"""
    
    # Append to log file
    try:
        with open(log_file, "a") as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Error writing to log file: {str(e)}")

def create_ui(config, theme_name="Ocean"):
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    .verification-passed {
        color: green;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(0, 255, 0, 0.1);
        display: inline-block;
        margin-top: 10px;
        text-align: center;
        width: 100%;
        font-size: 18px;
    }
    .verification-failed {
        color: red;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 0, 0, 0.1);
        display: inline-block;
        margin-top: 10px;
        text-align: center;
        width: 100%;
        font-size: 18px;
    }
    .verification-unclear {
        color: orange;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 165, 0, 0.1);
        display: inline-block;
        margin-top: 10px;
        text-align: center;
        width: 100%;
        font-size: 18px;
    }
    .verification-not-run {
        color: gray;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(128, 128, 128, 0.1);
        display: inline-block;
        margin-top: 10px;
        text-align: center;
        width: 100%;
        font-size: 18px;
    }
    .test-result-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        background-color: #f9f9f9;
    }
    .test-result-header {
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 16px;
        color: #333;
    }
    .test-details {
        margin-top: 15px;
        padding: 10px;
        background-color: white;
        border-radius: 5px;
        border-left: 4px solid #ccc;
        line-height: 1.5;
        white-space: pre-line;
        color: #333;
    }
    .test-details.passed {
        border-left-color: green;
    }
    .test-details.failed {
        border-left-color: red;
    }
    .test-details.unclear {
        border-left-color: orange;
    }
    .test-details strong {
        font-weight: bold;
        color: #000;
    }
    .analysis-section {
        margin-top: 10px;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        font-style: italic;
        color: #333;
    }
    """

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_map[theme_name], css=css
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value=config['agent_type'],
                        info="Select the type of agent to use",
                    )
                    with gr.Column():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=config['max_steps'],
                            step=1,
                            label="Max Run Steps",
                            info="Maximum number of steps the agent will take",
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=config['max_actions_per_step'],
                            step=1,
                            label="Max Actions per Step",
                            info="Maximum number of actions the agent will take per step",
                        )
                    with gr.Column():
                        use_vision = gr.Checkbox(
                            label="Use Vision",
                            value=config['use_vision'],
                            info="Enable visual processing capabilities",
                        )
                        tool_calling_method = gr.Dropdown(
                            label="Tool Calling Method",
                            value=config['tool_calling_method'],
                            interactive=True,
                            allow_custom_value=True,  # Allow users to input custom model names
                            choices=["auto", "json_schema", "function_calling"],
                            info="Tool Calls Funtion Name",
                            visible=False
                        )

            with gr.TabItem("🔧 LLM Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        choices=[provider for provider,model in utils.model_names.items()],
                        label="LLM Provider",
                        value=config['llm_provider'],
                        info="Select your preferred language model provider"
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        choices=utils.model_names['openai'],
                        value=config['llm_model_name'],
                        interactive=True,
                        allow_custom_value=True,  # Allow users to input custom model names
                        info="Select a model from the dropdown or type a custom model name"
                    )
                    llm_num_ctx = gr.Slider(
                        minimum=2**8,
                        maximum=2**16,
                        value=config['llm_num_ctx'],
                        step=1,
                        label="Max Context Length",
                        info="Controls max context length model needs to handle (less = faster)",
                        visible=config['llm_provider'] == "ollama"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config['llm_temperature'],
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value=config['llm_base_url'],
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=config['llm_api_key'],
                            info="Your API key (leave blank to use .env)"
                        )

            # Change event to update context length slider
            def update_llm_num_ctx_visibility(provider):
                """Show/hide the context length slider based on the provider."""
                return gr.update(visible=provider == "ollama")

            # Bind the change event of llm_provider to update the visibility of context length slider
            llm_provider.change(
                fn=update_llm_num_ctx_visibility,
                inputs=llm_provider,
                outputs=llm_num_ctx
            )

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=config['use_own_browser'],
                            info="Use your existing browser instance",
                        )
                        keep_browser_open = gr.Checkbox(
                            label="Keep Browser Open",
                            value=config['keep_browser_open'],
                            info="Keep Browser Open between Tasks",
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=config['headless'],
                            info="Run browser without GUI",
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=config['disable_security'],
                            info="Disable browser security features",
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=config['enable_recording'],
                            info="Enable saving browser recordings",
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=config['window_w'],
                            info="Browser window width",
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=config['window_h'],
                            info="Browser window height",
                        )


                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value=config['save_recording_path'],
                        info="Path to save browser recordings",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    chrome_cdp = gr.Textbox(
                        label="CDP URL",
                        placeholder="http://localhost:9222",
                        value="",
                        info="CDP for google remote debugging",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value=config['save_trace_path'],
                        info="Path to save Agent traces",
                        interactive=True,
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value=config['save_agent_history_path'],
                        info="Specify the directory where agent history should be saved.",
                        interactive=True,
                    )

            with gr.TabItem("🤖 Run Agent", id=4):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value=config['task'],
                    info="Describe what you want the agent to do",
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                    
                with gr.Row():
                    browser_view = gr.HTML(
                        value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
                        label="Live Browser View",
                )
            
            with gr.TabItem("🧐 Deep Research", id=5):
                research_task_input = gr.Textbox(label="Research Task", lines=5, value="Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects, substantiated with examples of relevant models and techniques. The report should reflect original insights and analysis, moving beyond mere summarization of existing literature.")
                with gr.Row():
                    max_search_iteration_input = gr.Number(label="Max Search Iteration", value=3, precision=0) # precision=0 确保是整数
                    max_query_per_iter_input = gr.Number(label="Max Query per Iteration", value=1, precision=0) # precision=0 确保是整数
                with gr.Row():
                    research_button = gr.Button("▶️ Run Deep Research", variant="primary", scale=2)
                    stop_research_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                markdown_output_display = gr.Markdown(label="Research Report")
                markdown_download = gr.File(label="Download Research Report")

            # Add Testing Tab
            with gr.TabItem("🧪 Testing", id=6):
                with gr.Group():
                    # Test Case Inputs
                    test_case_name = gr.Textbox(
                        label="Test Case Name",
                        placeholder="e.g., Verify Issue Title",
                        value="Untitled Test"
                    )
                    task_description = gr.Textbox(
                        label="Task Description",
                        lines=4,
                        placeholder="e.g., Navigate to https://github.com/repo/issues and find the 2nd issue"
                    )
                    verification_condition = gr.Textbox(
                        label="Verification Condition",
                        lines=2,
                        placeholder="e.g., Verify that the 2nd issue title is 'Fix Bug XYZ'"
                    )
                    
                    # Save/Load Test Case
                    with gr.Row():
                        save_test_button = gr.Button("💾 Save Test Case", variant="secondary")
                        load_test_dropdown = gr.Dropdown(
                            label="Load Saved Test Case",
                            choices=list_test_cases(),
                            type="value",
                            interactive=True
                        )
                        refresh_tests_button = gr.Button("🔄 Refresh", size="sm")
                        load_test_file = gr.File(
                            label="Or Upload Test Case File",
                            file_types=[".json"],
                            type="filepath"
                        )
                    
                    test_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=True
                    )

                    # LLM Configuration
                    with gr.Row():
                        llm_provider = gr.Dropdown(
                            label="LLM Provider",
                            choices=["openai", "anthropic", "ollama"],
                            value=config['llm_provider'],
                            info="Select the LLM provider to use"
                        )
                        llm_model_name = gr.Dropdown(
                            label="LLM Model",
                            choices=utils.model_names.get(config['llm_provider'], []),
                            value=config['llm_model_name'],
                            interactive=True,
                            allow_custom_value=True
                        )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config['llm_temperature'],
                        step=0.1,
                        label="Temperature",
                        info="Higher values make output more random, lower values more deterministic"
                    )
                    llm_num_ctx = gr.Slider(
                        minimum=256,
                        maximum=65536,
                        value=config['llm_num_ctx'],
                        step=1,
                        label="Context Length",
                        visible=config['llm_provider'] == "ollama",
                        info="Maximum context length for the model (only for Ollama)"
                    )

                    # Agent Settings
                    with gr.Row():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=config['max_steps'],
                            step=1,
                            label="Max Steps",
                            info="Maximum number of steps the agent will take"
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=config['max_actions_per_step'],
                            step=1,
                            label="Max Actions per Step",
                            info="Maximum number of actions the agent can take in a single step"
                        )
                    use_vision = gr.Checkbox(
                        label="Use Vision",
                        value=config['use_vision'],
                        info="Enable vision capabilities for the agent"
                    )
                    tool_calling_method = gr.Dropdown(
                        label="Tool Calling Method",
                        choices=["auto", "json_schema", "function_calling"],
                        value=config['tool_calling_method'],
                        info="Method used for tool calling"
                    )

                    # Advanced Settings (Collapsible)
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Group():
                            # Browser Settings
                            use_own_browser = gr.Checkbox(
                                label="Use Own Browser",
                                value=config['use_own_browser'],
                                info="Use your own browser instance"
                            )
                            keep_browser_open = gr.Checkbox(
                                label="Keep Browser Open",
                                value=config['keep_browser_open'],
                                info="Keep the browser open after the agent finishes"
                            )
                            headless = gr.Checkbox(
                                label="Headless Mode",
                                value=config['headless'],
                                info="Run the browser in headless mode (no UI)"
                            )
                            with gr.Row():
                                window_w = gr.Number(
                                    label="Window Width",
                                    value=config['window_w'],
                                    info="Browser window width"
                                )
                                window_h = gr.Number(
                                    label="Window Height",
                                    value=config['window_h'],
                                    info="Browser window height"
                                )
                            # Recording and Trace Paths
                            save_recording_path = gr.Textbox(
                                label="Recording Path",
                                placeholder="e.g., ./tmp/record_videos",
                                value=config['save_recording_path'],
                                info="Path to save browser recordings"
                            )
                            save_trace_path = gr.Textbox(
                                label="Trace Path",
                                placeholder="e.g., ./tmp/traces",
                                value=config['save_trace_path'],
                                info="Path to save browser traces"
                            )

                    # Run Test Button
                    run_test_button = gr.Button("Run Test", variant="primary")

                    # Outputs
                    with gr.Row():
                        verification_result = gr.Textbox(
                            label="Agent Output",
                            lines=10,
                            interactive=False,
                            placeholder="Agent output and verification result will appear here"
                        )
                        
                    pass_fail_badge = gr.HTML(
                        value='<div class="test-result-container"><div class="verification-not-run">⏱️ NOT RUN</div></div>'
                    )

            with gr.TabItem("📊 Results", id=7):
                with gr.Group():

                    recording_display = gr.Video(label="Latest Recording")

                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label="Final Result", lines=3, show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label="Errors", lines=3, show_label=True
                            )
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label="Model Actions", lines=3, show_label=True
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label="Model Thoughts", lines=3, show_label=True
                            )

                    trace_file = gr.File(label="Trace File")

                    agent_history_file = gr.File(label="Agent History")

                # Bind the stop button click event after errors_output is defined
                stop_button.click(
                    fn=stop_agent,
                    inputs=[],
                    outputs=[errors_output, stop_button, run_button],
                )

                # Run button click handler
                run_button.click(
                    fn=run_with_stream,
                        inputs=[
                            agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                            use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                            save_recording_path, save_agent_history_path, save_trace_path,  # Include the new path
                            enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step, tool_calling_method, chrome_cdp
                        ],
                    outputs=[
                        browser_view,           # Browser view
                        final_result_output,    # Final result
                        errors_output,          # Errors
                        model_actions_output,   # Model actions
                        model_thoughts_output,  # Model thoughts
                        recording_display,      # Latest recording
                        trace_file,             # Trace file
                        agent_history_file,     # Agent history file
                        stop_button,            # Stop button
                        run_button              # Run button
                    ],
                )
                
                # Run Deep Research
                research_button.click(
                        fn=run_deep_search,
                        inputs=[research_task_input, max_search_iteration_input, max_query_per_iter_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision, use_own_browser, headless, chrome_cdp],
                        outputs=[markdown_output_display, markdown_download, stop_research_button, research_button]
                )
                # Bind the stop button click event after errors_output is defined
                stop_research_button.click(
                    fn=stop_research_agent,
                    inputs=[],
                    outputs=[stop_research_button, research_button],
                )

            with gr.TabItem("🎥 Recordings", id=8):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []

                    # Get all video files
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

                    # Sort recordings by creation time (oldest first)
                    recordings.sort(key=os.path.getctime)

                    # Add numbering to the recordings
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))

                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    value=list_recordings(config['save_recording_path']),
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )
            
            with gr.TabItem("📁 Configuration", id=9):
                with gr.Group():
                    config_file_input = gr.File(
                        label="Load Config File",
                        file_types=[".pkl"],
                        interactive=True
                    )

                    load_config_button = gr.Button("Load Existing Config From File", variant="primary")
                    save_config_button = gr.Button("Save Current Config", variant="primary")

                    config_status = gr.Textbox(
                        label="Status",
                        lines=2,
                        interactive=False
                    )

                load_config_button.click(
                    fn=update_ui_from_config,
                    inputs=[config_file_input],
                    outputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security, enable_recording,
                        window_w, window_h, save_recording_path, save_trace_path, save_agent_history_path,
                        task, config_status
                    ]
                )

                save_config_button.click(
                    fn=save_current_config,
                    inputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security,
                        enable_recording, window_w, window_h, save_recording_path, save_trace_path,
                        save_agent_history_path, task,
                    ],  
                    outputs=[config_status]
                )

            # Run Test Button
            run_test_button.click(
                fn=run_verification_test,
                inputs=[
                    test_case_name, task_description, verification_condition,
                    llm_provider, llm_model_name, llm_temperature, llm_num_ctx,
                    max_steps, max_actions_per_step, use_vision, tool_calling_method,
                    use_own_browser, keep_browser_open, headless, window_w, window_h,
                    save_recording_path, save_trace_path
                ],
                outputs=[verification_result, pass_fail_badge]
            )
            
            # Save Test Case Button
            save_test_button.click(
                fn=lambda test_case_name, task_description, verification_condition: (
                    save_test_case(test_case_name, task_description, verification_condition)[0],  # Get status message
                    gr.update(choices=list_test_cases())  # Refresh dropdown
                ),
                inputs=[test_case_name, task_description, verification_condition],
                outputs=[test_status, load_test_dropdown]
            )
            
            # Load Test Case from File
            load_test_file.change(
                fn=load_test_case,
                inputs=[load_test_file],
                outputs=[test_status, test_case_name, task_description, verification_condition]
            )
            
            # Load Test Case from Dropdown
            def load_test_from_dropdown(selected_test):
                if not selected_test:
                    return "No test case selected", gr.update(), gr.update(), gr.update()
                
                try:
                    import json
                    with open(selected_test, "r") as f:
                        test_case = json.load(f)
                    
                    return (
                        f"Test case '{test_case['name']}' loaded successfully",
                        gr.update(value=test_case.get("name", "")),
                        gr.update(value=test_case.get("task_description", "")),
                        gr.update(value=test_case.get("verification_condition", ""))
                    )
                except Exception as e:
                    return f"Error loading test case: {str(e)}", gr.update(), gr.update(), gr.update()
            
            load_test_dropdown.change(
                fn=load_test_from_dropdown,
                inputs=[load_test_dropdown],
                outputs=[test_status, test_case_name, task_description, verification_condition]
            )
            
            # Refresh Test Cases Button
            refresh_tests_button.click(
                fn=lambda: (gr.update(choices=list_test_cases()), "Test case list refreshed"),
                inputs=[],
                outputs=[load_test_dropdown, test_status]
            )

        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            fn=update_model_dropdown,
            inputs=llm_provider,
            outputs=llm_model_name
        )

        # Add this after defining the components
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()

    demo = create_ui(config_dict, theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == '__main__':
    main()
