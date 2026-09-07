"""Hermes Finance: an importable, in-process personal finance domain module."""

from hermes_finance.config import FinanceConfig
from hermes_finance.corrections import edit_transaction_amount
from hermes_finance.domain import (
    Direction,
    Transaction,
    TransactionStatus,
    normalize_optional_text,
    normalize_required_text,
    normalize_usdt_amount,
    require_aware_datetime,
    require_calendar_date,
)
from hermes_finance.filtered_summary import (
    select_category_source,
    select_monthly_category,
)
from hermes_finance.ingest import (
    IngestConsistencyError,
    IngestDisposition,
    IngestResult,
    ingest_transaction,
)
from hermes_finance.integration import (
    get_monthly_category_summary_text,
    get_monthly_report_text,
    get_monthly_source_summary_text,
    get_recent_transactions_text,
    get_transactions_by_date_text,
    get_transactions_by_month_text,
    ingest_finance_message,
)
from hermes_finance.ledger import create_transaction
from hermes_finance.mutations import (
    TransactionMutationError,
    TransactionNotActiveError,
    TransactionNotFoundError,
    edit_transaction,
    soft_delete_transaction,
)
from hermes_finance.operations import (
    list_recent_transactions,
    list_transactions_by_date,
    list_transactions_by_month,
)
from hermes_finance.parser import (
    ParsedTransactionInput,
    TransactionParseError,
    parse_transaction_input,
)
from hermes_finance.periods import (
    DATE_PERIODS,
    MONTH_PERIODS,
    resolve_relative_date,
    resolve_relative_month,
)
from hermes_finance.provenance import (
    TelegramMessageIdentity,
    TelegramMessageRef,
    TelegramUpdateIdentity,
)
from hermes_finance.rendering import (
    render_monthly_category_summary,
    render_monthly_report,
    render_monthly_source_summary,
    render_recent_transactions,
    render_transactions_by_date,
    render_transactions_by_month,
)
from hermes_finance.reporting import (
    CategoryReport,
    MonthlyFinanceReport,
    SourceReport,
    build_monthly_report,
)
from hermes_finance.repository import (
    RepositoryDataError,
    RepositoryTransactionError,
    find_transaction_by_message,
    find_transaction_by_update,
    get_processed_update_ref,
    get_transaction,
    is_update_processed,
    persist_transaction,
    record_processed_update,
)
from hermes_finance.storage import (
    SCHEMA_VERSION,
    DatabaseMigrationError,
    open_database,
)

__version__ = "0.1.0"

__all__ = [
    "DATE_PERIODS",
    "MONTH_PERIODS",
    "SCHEMA_VERSION",
    "CategoryReport",
    "DatabaseMigrationError",
    "Direction",
    "FinanceConfig",
    "IngestConsistencyError",
    "IngestDisposition",
    "IngestResult",
    "MonthlyFinanceReport",
    "ParsedTransactionInput",
    "RepositoryDataError",
    "RepositoryTransactionError",
    "SourceReport",
    "TelegramMessageIdentity",
    "TelegramMessageRef",
    "TelegramUpdateIdentity",
    "Transaction",
    "TransactionMutationError",
    "TransactionNotActiveError",
    "TransactionNotFoundError",
    "TransactionParseError",
    "TransactionStatus",
    "build_monthly_report",
    "create_transaction",
    "edit_transaction",
    "edit_transaction_amount",
    "find_transaction_by_message",
    "find_transaction_by_update",
    "get_monthly_category_summary_text",
    "get_monthly_report_text",
    "get_monthly_source_summary_text",
    "get_processed_update_ref",
    "get_recent_transactions_text",
    "get_transaction",
    "get_transactions_by_date_text",
    "get_transactions_by_month_text",
    "ingest_finance_message",
    "ingest_transaction",
    "is_update_processed",
    "list_recent_transactions",
    "list_transactions_by_date",
    "list_transactions_by_month",
    "normalize_optional_text",
    "normalize_required_text",
    "normalize_usdt_amount",
    "open_database",
    "parse_transaction_input",
    "persist_transaction",
    "record_processed_update",
    "render_monthly_category_summary",
    "render_monthly_report",
    "render_monthly_source_summary",
    "render_recent_transactions",
    "render_transactions_by_date",
    "render_transactions_by_month",
    "require_aware_datetime",
    "require_calendar_date",
    "resolve_relative_date",
    "resolve_relative_month",
    "select_category_source",
    "select_monthly_category",
    "soft_delete_transaction",
]
