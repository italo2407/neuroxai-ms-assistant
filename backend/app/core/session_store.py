import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch


@dataclass
class SessionData:
    session_id: str
    image_tensor: Optional[torch.Tensor] = None
    image_np: Optional[np.ndarray] = None         # original (H,W) float32 [0,1]
    image_original_size: tuple = (224, 224)
    gt_mask_np: Optional[np.ndarray] = None       # (H,W) binary float32
    predicted_mask_np: Optional[np.ndarray] = None
    soft_logits_np: Optional[np.ndarray] = None
    fold_masks: list = field(default_factory=list)
    xai_results: dict = field(default_factory=dict)    # method -> float32 map (H,W)
    vlg_cbm_concepts: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self):
        self.updated_at = time.time()


class SessionStore:
    def __init__(self, ttl_minutes: int = 30):
        self._store: dict[str, SessionData] = {}
        self._ttl = ttl_minutes * 60
        self._lock = asyncio.Lock()

    def create(self) -> SessionData:
        sid = str(uuid.uuid4())
        session = SessionData(session_id=sid)
        self._store[sid] = session
        return session

    def get(self, session_id: str) -> Optional[SessionData]:
        return self._store.get(session_id)

    def update(self, session: SessionData):
        session.touch()
        self._store[session.session_id] = session

    async def cleanup_expired(self):
        """Background task to evict expired sessions."""
        while True:
            await asyncio.sleep(300)  # check every 5 minutes
            now = time.time()
            expired = [sid for sid, s in self._store.items() if now - s.updated_at > self._ttl]
            for sid in expired:
                del self._store[sid]


session_store = SessionStore()
