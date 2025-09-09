import asyncio
from agents.assistant_agent import create_assistant_agent

async def main():
    print(">>")
    assistant = await create_assistant_agent()

    while True:
        user_input = input("🤖 Ask the assistant something (type 'exit' to quit):\n> ")
        if user_input.lower() in ("exit", "quit"):
            break

        # Let the assistant handle the task
        response = await assistant.ask(user_input)
        print(f"\n💬 Assistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
