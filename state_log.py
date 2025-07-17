from langchain_core.messages import BaseMessage


def dump_to_file(content: str, filename: str):
    with open(filename, "w") as f:
        f.write(content)


def dump_messages_to_file(messages: list[BaseMessage]):
    messages_str = "\n".join([msg.pretty_repr() for msg in messages])
    dump_to_file(messages_str, "out/messages.dump.txt")


def dump_conversation_history_to_file(state: dict):
    conv_history_str_parts = []
    for user in state["conversations"].keys():
        conv_history_str_parts.append(f"<{user}>")
        for msg in state["conversations"][user]:
            conv_history_str_parts.append(f"{msg}")
        conv_history_str_parts.append(f"</{user}>")
        conv_history_str_parts.append(f"\n")
    conv_history_str = "\n".join(conv_history_str_parts)
    dump_to_file(conv_history_str, "out/conv_history.dump.txt")
