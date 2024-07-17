from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Sequence
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory

class CustomChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = Field(default_factory=list)
    max_messages: int = 4  # Set the limit (K) of messages to keep

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)
        # Keep last k messages
        self.messages = self.messages[-self.max_messages:]


    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the store"""
        self.add_messages(messages)

    def clear(self) -> None:
        self.messages = []

    async def aclear(self) -> None:
        self.clear()


class ChatStore:
    """A store for managing chat histories using Redis Cloud."""

    def _init_(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password
        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password
        )

    def get_session_history(self, user_id: str, conversation_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=f"{user_id}:{conversation_id}",
            key_prefix="chat_history",
            url=f"redis://:{self.password}@{self.host}:{self.port}",
            ttl=3600  # Time to live in seconds, adjust as needed
        )
    
    def clear_session_history(self, user_id: str, conversation_id: str) -> None:
        """Clear the session history for a given user and conversation."""
        history = self.get_session_history(user_id, conversation_id)
        history.clear()

    def add_message_to_history(self, user_id: str, conversation_id: str, message: BaseMessage) -> None:
        """Add a message to the session history for a given user and conversation."""
        history = self.get_session_history(user_id, conversation_id)
        history.add_message(message)

    # ... [rest of the methods remain the same] ...
