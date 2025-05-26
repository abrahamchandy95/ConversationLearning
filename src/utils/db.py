from typing import List, Optional, Dict, Any, Set
import pandas as pd
from supabase import Client
from postgrest.base_request_builder import APIResponse


def build_query(
    client: Client,
    table: str,
    thread_ids: Optional[List[str]] = None
) -> Any:
    """
    Builds a query with an optional filter
    """
    q = client.table(table).select('*')
    if thread_ids:
        q = q.in_("thread_id", thread_ids)
    return q


def get_total_rows(
    client: Client, table: str, filters: Optional[List[str]] = None
) -> int:
    """
    Return the exact row count for `table` using PostgREST count=exact.
    """
    query = client.table(table).select("*", count="exact").limit(1)
    if filters:
        query = query.in_("thread_id", filters)
    resp: APIResponse = query.execute()
    return resp.count or 0


def convert_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df.sort_values(['thread_id', 'created_at'], inplace=True)
    return df


def fetch_batch(
    client,
    table: str,
    offset: int = 0,
    batch_size: int = 1000,
    filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch up to `batch_size` rows from `table`, starting at `offset`,
    selecting exactly `select_str`.
    """
    query = build_query(client, table, filter)
    query = query.limit(batch_size).offset(offset)
    resp: APIResponse = query.execute()
    return resp.data or []


def fetch_all_batches(
    client: Client,
    table: str,
    thread_ids: Optional[List[str]] = None,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Fetch *all* rows from `table` in increments of `batch_size`.
    Returns a list of row‑dicts.
    """
    total = get_total_rows(client, table, thread_ids)
    all_rows: List[Dict[str, Any]] = []
    # iterate upto supabase's limit of batch size
    for offset in range(0, total, batch_size):
        batch = fetch_batch(client, table, offset, batch_size, thread_ids)
        if not batch:
            break
        all_rows.extend(batch)
    return convert_to_dataframe((all_rows))


def load_thread_ids(
    client: Client,
    table: str = 'chat_messages',
    batch_size: int = 1000
) -> Set[str]:
    """
    Return the set of distinct thread_id values in `table`.
    """
    thread_ids: Set[str] = set()
    offset = 0

    while True:
        response = (
            client.table(table).select("thread_id")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        data = response.data or []
        thread_ids.update(row["thread_id"] for row in data)
        if len(data) < batch_size:
            break
        offset += batch_size
    return thread_ids


def load_conversation(
    client: Client,
    thread_id: str
) -> pd.DataFrame:
    """
    Load all messages for one `thread_id`, sorted by created_at.
    """
    return fetch_all_batches(
        client, table="chat_messages", thread_ids=[thread_id]
    )


def load_chats(
    client: Client,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Load all chat_messages in batches, flatten nested chat_threads.user_id,
    and return a DataFrame sorted by thread_id and created_at.
    """
    return fetch_all_batches(
        client, table="chat_messages", batch_size=batch_size
    )


def load_chats_from_threads(
    client: Client,
    thread_ids: List[str],
    batch_size: int = 1000
) -> pd.DataFrame:
    """Loads chat messages from multiple threads"""
    return fetch_all_batches(
        client,
        table="chat_messages",
        thread_ids=thread_ids,
        batch_size=batch_size
    )
