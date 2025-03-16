import pdb

import pyperclip
from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging
import json

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            # send text to browser
            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)
            
        @self.registry.action("Get element attributes")
        async def get_element_attributes(browser: BrowserContext, index: int, attributes: List[str]):
            """
            Retrieve specified attributes for an element by its data-browser-use-index.
            
            Args:
                browser (BrowserContext): The browser context.
                index (int): The index of the element to inspect.
                attributes (List[str]): List of attribute names to retrieve (e.g., ['href', 'disabled']).
            
            Returns:
                ActionResult: A dictionary of attribute names and their values, or an error if the element is not found.
            """
            page = await browser.get_current_page()
            element = await page.query_selector(f"[data-browser-use-index='{index}']")
            if not element:
                return ActionResult(error=f"Element with index {index} not found.")
            
            attr_dict = {}
            for attr in attributes:
                value = await element.get_attribute(attr)
                attr_dict[attr] = value if value is not None else "not present"
                
            # For inner_text, we need to use a different method
            if 'inner_text' in attributes:
                inner_text = await element.inner_text()
                attr_dict['inner_text'] = inner_text
                
            # For tag_name, we need to use evaluate
            if 'tag_name' in attributes:
                tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                attr_dict['tag_name'] = tag_name
                
            return ActionResult(extracted_content=json.dumps(attr_dict))
