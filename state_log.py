from langchain_core.messages import BaseMessage


def dump_to_file(content: str, filename: str):
    with open(filename, "w") as f:
        f.write(content)


def dump_messages_to_file(messages: list[BaseMessage]):
    messages_str = "\n".join([msg.pretty_repr() for msg in messages])
    dump_to_file(messages_str, "out/messages.dump.txt")
