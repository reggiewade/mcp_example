import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import json
import sys

class MCPClient:
    def __init__(self, model: str = "minimax-m3"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model = model

        # Point OpenAI client at local Ollama server
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required by client but unused by Ollama
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"

        # build config describing how to launch MCP
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # spawns the server as a subprocess and opens communication over stdin/stdout
        # exit_stack ensures that the subprocess closes when done, even in the case of an error
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        # separates transport into two streams
        self.stdio, self.write = stdio_transport
        # ClientSession gives structure to MCP interface
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # perform MCP initialization
        await self.session.initialize()

        # List available tools (query server)
        res = await self.session.list_tools()
        tools = res.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in response.tools
        ]

        # Initial Ollama API call
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        # Only break the while loop when there are no more tool calls
        while True:
            message = response.choices[0].message

            # Handle text content
            if message.content:
                final_text.append(message.content)

            # Handle tool calls
            if not message.tool_calls:
                break  # No more tool calls, we're done

            for tool_call in message.tool_calls: 
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Append assistant message and tool result
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                })

            # Get next response
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python client.py <path_to_server_script>")
            sys.exit(1)

        client = MCPClient()
        try:
            await client.connect_to_server(sys.argv[1])
            await client.chat_loop()
        finally:
            await client.cleanup()

    if __name__ == "__main__":
        import sys
        asyncio.run(main())