"""Efficient Queries made to supabase"""
import time
from typing import Optional, List, Dict, Set, Any
from urllib.error import HTTPError
import pandas as pd
from supabase import Client
from postgrest.base_request_builder import APIResponse


class TableFetcher:
    """
    Class to effectively query from supabase in batches
    """

    def __init__(
        self,
        client: Client,
        table: str,
        page_size: int = 100,
    ):
        self.client = client
        self.table = table
        self.page_size = page_size
        self.max_retries = 3
        self.backoff_seconds = 0.5
        self.thread_batch_size = 100

        # cursor state
        self._last_seen: Optional[Any] = None

    def fetch_all(
        self,
        thread_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        If thread_ids is provided, split it into chunks of thread_batch_size,
        and for each chunk do a fresh cursor‐based scan. Collate all rows.
        If no thread_ids, do one scan of the whole table.
        """
        all_rows: List[Dict[str, Any]] = []

        if thread_ids:
            # split the IDs into manageable batches
            for start in range(0, len(thread_ids), self.thread_batch_size):
                chunk = thread_ids[start: start + self.thread_batch_size]
                # reset cursor for this chunk
                self._last_seen = None

                # page through just these IDs
                while True:
                    batch = self._retry_batch_fetches(chunk)
                    if not batch:
                        break
                    all_rows.extend(batch)
                    self._last_seen = batch[-1]["created_at"]
        else:
            # no filtering: one full scan
            self._last_seen = None
            while True:
                batch = self._retry_batch_fetches(None)
                if not batch:
                    break
                all_rows.extend(batch)
                self._last_seen = batch[-1]["created_at"]

        # final assemble
        df = pd.DataFrame(all_rows)
        if not df.empty and "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            df.sort_values(["thread_id", "created_at"], inplace=True)
        return df

    def fetch_page(
        self,
        thread_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetches one page of rows from the last cursor postion"""
        batch = self._retry_batch_fetches(thread_ids)
        if batch:
            self._last_seen = batch[-1]["created_at"]
        return batch

    def _fetch_next_batch(
        self,
        thread_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        # build query with ordering
        query = (
            self.client
            .table(self.table)
            .select("*")
            .order("created_at")
        )

        # look after the last seen item in the column
        if self._last_seen is not None:
            query = query.gt("created_at", self._last_seen)

        # batched thread_id filter
        if thread_ids:
            query = self._filter_threads_in_batches(query, thread_ids)

        # limit page
        query = query.limit(self.page_size)

        resp: APIResponse = query.execute()
        return resp.data or []

    def _filter_threads_in_batches(
        self,
        query,
        thread_ids: List[str]
    ):
        """Helps filtering queries in batches to reduce load"""
        or_clauses = []
        for i in range(0, len(thread_ids), self.thread_batch_size):
            batch = thread_ids[i:i+self.thread_batch_size]
            or_clauses.append(f"thread_id.in.({','.join(batch)})")

        # Combine with OR operator
        return query.or_(",".join(or_clauses))

    def _retry_batch_fetches(
        self,
        thread_ids: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Wrap `_fetch_next_batch` in retries with exponential backoff.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._fetch_next_batch(thread_ids)
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_seconds * attempt)
        return []


def load_thread_ids(
    client: Client,
    table: str = 'chat_messages',
    batch_size: int = 1000
) -> Set[str]:
    """
    Return a set of all thread_id values in `table`, batching
    in chunks of `batch_size` to respect Supabase's limit.
    """
    thread_ids: Set[str] = set()
    offset = 0

    while True:
        try:
            resp: APIResponse = (
                client.table(table)
                .select("thread_id")
                .range(offset, offset + batch_size - 1)
                .execute()
            )
        except HTTPError:
            break
        rows = resp.data or []
        # add non-null thread_id values
        thread_ids.update(r["thread_id"]
                          for r in rows if r.get("thread_id") is not None)
        if len(rows) < batch_size:
            break
        offset += len(rows)

    return thread_ids


def fetch_new_thread_ids(
    client: Client
) -> Set[str]:
    """
    Helper that returns thread_ids in chat_messages not yet in chat_inference
    """
    all_ids = load_thread_ids(client, table="chat_messages")
    try:
        done_ids = load_thread_ids(client, table="chat_inference")
    except HTTPError:
        done_ids = set()
    return all_ids - done_ids


def load_chats_from_threads(
    client: Client,
    thread_ids: List[str],
    **fetcher_kwargs
) -> pd.DataFrame:
    """
    Convenience wrapper: uses TableFetcher to page through chat_messages
    for the given list of thread_ids.
    Any additional keyword args are passed to TableFetcher.__init__.
    """
    fetcher = TableFetcher(client, table="chat_messages", **fetcher_kwargs)
    return fetcher.fetch_all(thread_ids=thread_ids)


def load_conversation(client: Client, thread_id: str) -> pd.DataFrame:
    """
    Back-compat single-thread loader.
    Wraps load_chats_from_threads, turning one ID into a one‐element list.
    """
    return (
        load_chats_from_threads(client, [thread_id])
        .sort_values("created_at")
    )
