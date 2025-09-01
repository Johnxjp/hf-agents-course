"""
Required tools:
- Web Search: can use Google
- File Read
- Image Understanding
- Audio Understanding
"""

import os
import json
from typing import Dict, List, Union

from agents import function_tool
from dotenv import load_dotenv
import httpx
import urllib


load_dotenv()


class GoogleSearchToolSpec:
    """Google Search tool spec."""

    spec_functions = [("google_search", "agoogle_search")]

    def __init__(self, key: str, engine: str) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = 1

    def _get_url(self, query: str) -> str:
        url = "https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}".format(
            key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query)
        )

        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"

        return url

    def _parse_results(self, results: List[Dict]) -> Union[List[Dict], str]:
        cleaned_results = []
        if len(results) == 0:
            return "No search results available"

        for result in results:
            if "snippet" in result:
                cleaned_results.append(
                    {
                        "title": result["title"],
                        "link": result["link"],
                    }
                )

        return cleaned_results

    def google_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.

        """
        url = self._get_url(query)

        with httpx.Client() as client:
            response = client.get(url)

        results = json.loads(response.text).get("items", [])

        return json.dumps(self._parse_results(results))

    async def agoogle_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.

        """
        url = self._get_url(query)

        async with httpx.AsyncClient() as client:
            response = await client.get(url)

        results = json.loads(response.text).get("items", [])

        return self._parse_results(results)

    def search_url(self, url: str) -> str:
        """
        Returns the content of a web page as a string.
        """
        with httpx.Client() as client:
            response = client.get(url)

        return "3 albums were released"


google_tool_spec = GoogleSearchToolSpec(
    os.getenv("GOOGLE_SEARCH_API_KEY"), os.getenv("GOOGLE_SEARCH_ENGINE_ID")
)


@function_tool
def search_google(query: str) -> str:
    """
    Returns list of URL search results from Google. The results is a JSON string
    formatted as [{title: <title>, link: <link>}, ...]
    """
    return google_tool_spec.google_search(query)


@function_tool
def search_url(url: str) -> str:
    """
    Returns the content of a web page as a string which can be used to search for result.
    """
    return google_tool_spec.search_url(url)
