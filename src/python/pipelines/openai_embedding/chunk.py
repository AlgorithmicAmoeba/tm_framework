import dataclasses
from typing import Any, Optional


@dataclasses.dataclass
class Chunk:
    """A chunk of text to embed."""
    document_hash: str = ""
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_hash": self.document_hash,
            "embedding": self.embedding
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        return cls(
            document_hash=data.get("document_hash", ""),
            embedding=data.get("embedding")
        )