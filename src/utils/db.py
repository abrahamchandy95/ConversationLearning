from typing import List, Optional, Dict, Any, Union, Set
import pandas as pd
from supabase import Client
from postgrest._sync.request_builder import SyncRequestBuilder


def build_query(
    client: Client,
    table: str,
    thread_ids: Optional[List[str]] = None
) -> SyncRequestBuilder:
    """
    Builds a query with an optional filter
    """
    query = client.table(table)
    if thread_ids:
        query = query.in_('thread_id', thread_ids)
    return query


def get_total_rows(
    client: Client, table: str, filters: Optional[List[str]] = None
) -> int:
    """
    Return the exact row count for `table` using PostgREST count=exact.
    """
    query = build_query(client, table, filters)
    resp = query.select('*', count='exact').execute()  # type: ignore
    return resp.count or 0


def fetch_batch(
    query: SyncRequestBuilder,
    select_str: str,
    offset: int,
    batch_size: int
) -> List[Dict[str, Any]]:
    """
    Fetch up to `batch_size` rows from `table`, starting at `offset`,
    selecting exactly `select_str`.
    """
    resp = (
        query.select(select_str)
        .range(offset, offset + batch_size - 1)
        .execute()
    )
    return resp.data or []


def fetch_all_batches(
    client: Client,
    table: str,
    select: Union[str, List[str]] = '*',
    thread_ids: Optional[List[str]] = None,
    batch_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetch *all* rows from `table` in increments of `batch_size`.
    Returns a list of row‑dicts.
    """
    if isinstance(select, list):
        select_str = ','.join(select)
    else:
        select_str = select
    total_rows = get_total_rows(client, table, thread_ids)
    rows: List[Dict[str, Any]] = []
    query = build_query(client, table, thread_ids)
    for offset in range(0, total_rows, batch_size):
        batch = fetch_batch(query, select_str, offset, batch_size)
        if not batch:
            break
        rows.extend(batch)
    return rows


def convert_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df.sort_values(['thread_id', 'created_at'], inplace=True)
    return df


def load_thread_ids(
    client: Client,
    table: str = 'chat_messages',
    batch_size: int = 1000
) -> Set[str]:
    """
    Return the set of distinct thread_id values in `table`.
    """
    rows = fetch_all_batches(
        client, table, select='thread_id', batch_size=batch_size
    )
    return {r['thread_id'] for r in rows if r.get('thread_id')}


def load_conversation(
    client: Client,
    thread_id: str
) -> pd.DataFrame:
    """
    Load all messages for one `thread_id`, sorted by created_at.
    """
    rows = fetch_all_batches(
        client,
        table='chat_messages',
        select='*',
        thread_ids=[thread_id]
    )
    return convert_to_dataframe(rows)


def load_chats(
    client: Client,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Load all chat_messages in batches, flatten nested chat_threads.user_id,
    and return a DataFrame sorted by thread_id and created_at.
    """
    rows = fetch_all_batches(
        client,
        table='chat_messages',
        select='*',
        batch_size=batch_size
    )
    return convert_to_dataframe(rows)


def load_chats_from_threads(
    client: Client,
    thread_ids: List[str],
    batch_size: int = 1000
) -> pd.DataFrame:
    """Loads chat messages from multiple threads"""
    rows = fetch_all_batches(
        client,
        table='chat_messages',
        select="*",
        thread_ids=thread_ids,
        batch_size=batch_size
    )
    return convert_to_dataframe(rows)
