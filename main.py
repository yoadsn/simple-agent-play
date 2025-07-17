import asyncio
import uuid
from collections import defaultdict
from email import message
from encodings import raw_unicode_escape
from multiprocessing import Value
from typing import Annotated

import click
from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, interrupt

from checpoint_storage import get_checkpointer, setup_checkpointer_db
from llm import ModelName, get_open_router_chat_model
from state_log import dump_conversation_history_to_file, dump_messages_to_file
from utils import run_async

llm = get_open_router_chat_model(ModelName.OR_GEMINI_2_0_FLASH_MODEL_NAME)


@tool
async def send_message(
    recipient: str, message: str, state: Annotated[dict, InjectedToolArg]
) -> str:
    """Send a message to a person."""
    message_from_ai = f"you->{recipient}: {message}"
    state["conversations"][recipient].append(message_from_ai)
    print(message_from_ai)
    return f"Message sent to {recipient}"


tools = [send_message]
tools_by_name = {tool.name: tool for tool in tools}


@task
def dump_messages(messages: list[BaseMessage]):
    dump_messages_to_file(messages)


@task
def dump_conversation_history(state: dict):
    dump_conversation_history_to_file(state)


async def get_user_message() -> HumanMessage:
    """Get user message."""
    user_message_request = interrupt({"action": "get_user_message"})
    if "from" not in user_message_request:
        return None
    return user_message_request


@task
def call_model(messages) -> BaseMessage:
    """Call model with a sequence of messages."""
    response = llm.bind_tools(tools).invoke(messages)
    return response


@task
async def call_tool(tool_call, state: dict, config: RunnableConfig):
    tool = tools_by_name[tool_call["name"]]
    args = {**tool_call["args"], "config": config, "state": state}
    observation = await tool.ainvoke(args)
    return {
        "tool_message": ToolMessage(content=observation, tool_call_id=tool_call["id"]),
        "next_state": state,
    }


def get_next_llm_input(this_user, this_message, conversations):
    conversation_strs = []
    for user in conversations.keys():
        all_messages_in_convo = "\n".join(conversations[user])
        conversation_with_user = f"""
<{user}>
{all_messages_in_convo}
</{user}>
"""
        conversation_strs.append(conversation_with_user)

    conversations_str = "\n".join(conversation_strs)

    next_llm_input = f"""Here are all the active conversations:
{conversations_str or "No conversations yet."}
You got a message from {this_user}:
---
{this_message}
---
Please reply.
"""

    return next_llm_input


# entrypoint
async def run_agent(
    input_state: dict[any, any], previous: dict[any, any], config: RunnableConfig
):
    state = {}

    system_message = SystemMessage(
        content=f"""You are speaking to multiple users at the same time.
You always have all history of conversations so you can reply in continuation to them.
If you reply to the user - you must use the {send_message.name} tool.
"""
    )

    state["conversations"] = {}

    while True:
        await dump_conversation_history(state)

        # Get user message
        user_message_request = await get_user_message()
        if not user_message_request:
            break

        this_user = user_message_request["from"]
        this_message = user_message_request["message"]

        conversations = state["conversations"]
        if this_user not in conversations:
            conversations[this_user] = []

        next_llm_input = get_next_llm_input(this_user, this_message, conversations)

        this_user_messages = conversations[this_user]
        this_user_messages.append(f"{this_user}->You: {this_message}")

        messages = [system_message, HumanMessage(content=next_llm_input)]

        await dump_messages(messages)

        llm_response = await call_model(messages)
        while True:
            if not llm_response.tool_calls:
                break

            # Execute tools
            tool_results = []
            for tool_call in llm_response.tool_calls:
                call_results = await call_tool(tool_call, state)
                tool_result = call_results["tool_message"]
                tool_results.append(tool_result)
                state = call_results["next_state"]

            # Append to message list
            new_messages = [llm_response, *tool_results]
            # for msg in new_messages:
            #     print(msg.pretty_print())
            messages = add_messages(messages, new_messages)

            # Call model again
            llm_response = await call_model(messages)

    return "done"


def get_input_for_interrupt(interrupt: Interrupt):
    next_input = None
    if interrupt.resumable:
        if interrupt.value["action"] == "get_user_message":
            user_from = input("Who: ")
            user_message = input("Msg: ")
            if not user_from or not user_message:
                next_input = Command(resume={"end": True})
            else:
                next_input = Command(
                    resume={"from": user_from, "message": user_message}
                )

        else:
            raise ValueError(f"Unknown interrupt action: {interrupt.value}")
    return next_input


@click.command()
@click.option(
    "--thread_id", default=None, help="Thread ID to run - or will use a random one"
)
@run_async
async def start(thread_id: str):
    next_input = {}
    thread_id = thread_id or str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    async with get_checkpointer() as checkpointer:
        runnable_pregel = entrypoint(checkpointer)(run_agent)

        while True:
            last_state = await runnable_pregel.aget_state(config)
            if last_state.interrupts:
                first_interrupt: Interrupt = last_state.interrupts[0]
                next_input = get_input_for_interrupt(first_interrupt)
            result = await runnable_pregel.ainvoke(next_input, config)
            if result == "done":
                break


@click.command()
@run_async
async def setup():
    await setup_checkpointer_db()


@click.group()
def cli():
    pass


cli.add_command(start)
cli.add_command(setup)

if __name__ == "__main__":
    cli()
