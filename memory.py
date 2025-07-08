import json
from redis import Redis
from typing import List, Dict

SESSION_KEY_TEMPLATE = "session:{bot_id}"

def save_message(bot_id: str, user_message: str, bot_response: str, redis_client: Redis):
    key = SESSION_KEY_TEMPLATE.format(bot_id=bot_id)
    entry = {"user": user_message, "bot": bot_response}
    redis_client.rpush(key, json.dumps(entry))
    redis_client.expire(key, 86400)

def load_session(bot_id: str, redis_client: Redis, limit: int = 20) -> List[Dict]:
    key = SESSION_KEY_TEMPLATE.format(bot_id=bot_id)
    entries = redis_client.lrange(key, -limit, -1)
    session = []
    for e in entries:
        try:
            session.append(json.loads(e))
        except:
            continue
    return session
