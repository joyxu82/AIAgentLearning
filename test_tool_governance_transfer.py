from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool_governance_demo_transfer import (  # noqa: E402
    ACCOUNTS,
    ApprovalStore,
    ExecutionContext,
    PermissionMode,
    ToolCall,
    TransferArgs,
    build_governance,
)

TRANSFER_ARGUMENTS = {
    "from_account": "ACC-A-123456",
    "to_account": "ACC-A-654321",
    "amount": 1000.0,
}


@pytest.fixture(autouse=True)
def isolated_accounts() -> Iterator[None]:
    # 每个用例都从预设余额开始，避免转账副作用在用例之间串味。
    snapshot = dict(ACCOUNTS)
    yield
    ACCOUNTS.clear()
    ACCOUNTS.update(snapshot)


def base_context(**overrides: object) -> ExecutionContext:
    context = ExecutionContext(
        trace_id="trace_transfer",
        user_id="u_100",
        tenant_id="tenant_a",
        permissions=frozenset({"transfer:create"}),
        allowed_tools=frozenset({"transfer"}),
    )
    return replace(context, **overrides) if overrides else context


def approve_transfer(
    approvals: ApprovalStore,
    approval_id: str,
    context: ExecutionContext,
    arguments: dict[str, object],
) -> None:
    approvals.approve(
        approval_id,
        user_id=context.user_id,
        tenant_id=context.tenant_id,
        tool_name="transfer",
        args=TransferArgs.model_validate(arguments),
    )


@pytest.mark.asyncio
async def test_transfer_invalid_arguments_never_debit() -> None:
    runtime, _, _ = build_governance()
    context = base_context()
    invalid_cases = [
        {**TRANSFER_ARGUMENTS, "approved": True, "user_id": "u_100"},
        {**TRANSFER_ARGUMENTS, "from_account": "ACC-1-12345"},
        {**TRANSFER_ARGUMENTS, "to_account": "acc-a-654321"},
        {**TRANSFER_ARGUMENTS, "amount": 0},
        {**TRANSFER_ARGUMENTS, "amount": 100_000.01},
    ]
    before = dict(ACCOUNTS)

    for index, arguments in enumerate(invalid_cases):
        result = await runtime.invoke(ToolCall(f"invalid_{index}", "transfer", arguments), context)
        assert result.error_code == "INVALID_ARGUMENT"

    assert ACCOUNTS == before


@pytest.mark.asyncio
async def test_transfer_precheck_blocks_limit_and_insufficient_balance() -> None:
    runtime, _, _ = build_governance()
    context = base_context()
    before = dict(ACCOUNTS)

    exceeded = await runtime.invoke(
        ToolCall("precheck_limit", "transfer", {**TRANSFER_ARGUMENTS, "amount": 60000.0}),
        context,
    )
    insufficient = await runtime.invoke(
        ToolCall(
            "precheck_balance",
            "transfer",
            {**TRANSFER_ARGUMENTS, "from_account": "ACC-A-654321", "amount": 10000.0},
        ),
        context,
    )

    assert exceeded.error_code == "EXCEED_LIMIT"
    assert insufficient.error_code == "INSUFFICIENT_BALANCE"
    assert ACCOUNTS == before


@pytest.mark.asyncio
async def test_transfer_permission_and_mode_limits_ignore_approval() -> None:
    runtime, approvals, _ = build_governance()
    context = base_context()
    approve_transfer(approvals, "approval_scope", context, TRANSFER_ARGUMENTS)
    approved = replace(context, approval_id="approval_scope")
    before = dict(ACCOUNTS)

    no_permission = await runtime.invoke(
        ToolCall("scope_permission", "transfer", TRANSFER_ARGUMENTS),
        replace(approved, permissions=frozenset()),
    )
    not_whitelisted = await runtime.invoke(
        ToolCall("scope_whitelist", "transfer", TRANSFER_ARGUMENTS),
        replace(approved, allowed_tools=frozenset({"transfer_disabled"})),
    )
    plan_mode = await runtime.invoke(
        ToolCall("scope_plan", "transfer", TRANSFER_ARGUMENTS),
        replace(approved, mode=PermissionMode.PLAN),
    )

    assert no_permission.error_code == "PERMISSION_DENIED"
    assert not_whitelisted.error_code == "TOOL_NOT_ALLOWED"
    assert plan_mode.error_code == "PLAN_MODE_DENIED"
    assert ACCOUNTS == before


@pytest.mark.asyncio
async def test_transfer_approval_binding_and_replay_protection() -> None:
    runtime, approvals, audit = build_governance()
    context = base_context()
    before = dict(ACCOUNTS)

    unapproved = await runtime.invoke(ToolCall("approval_missing", "transfer", TRANSFER_ARGUMENTS), context)
    assert unapproved.error_code == "APPROVAL_REQUIRED"
    assert ACCOUNTS == before

    # 审批绑定的是规范化参数摘要：先按 2000 元审批，再用 1000 元调用必须失效。
    approve_transfer(approvals, "approval_changed", context, {**TRANSFER_ARGUMENTS, "amount": 2000.0})
    changed = await runtime.invoke(
        ToolCall("approval_amount_changed", "transfer", TRANSFER_ARGUMENTS),
        replace(context, approval_id="approval_changed"),
    )
    assert changed.error_code == "APPROVAL_REQUIRED"
    assert ACCOUNTS == before

    approve_transfer(approvals, "approval_ok", context, TRANSFER_ARGUMENTS)
    approved_context = replace(context, approval_id="approval_ok", idempotency_key="transfer_ok_123456")
    succeeded = await runtime.invoke(
        ToolCall("transfer_ok_123456", "transfer", TRANSFER_ARGUMENTS),
        approved_context,
    )

    assert succeeded.ok is True
    assert succeeded.content["from"] == "ACC-A-****3456"
    assert succeeded.content["to"] == "ACC-A-****4321"
    assert succeeded.content["txn_id"] == "txn_123456"
    assert succeeded.content["status"] == "accepted"
    assert ACCOUNTS[("tenant_a", "ACC-A-123456")] == 99000.0
    assert ACCOUNTS[("tenant_a", "ACC-A-654321")] == 6000.0
    assert audit.records[-1]["ok"] is True
    assert "ACC-A-****3456" in json.dumps(audit.records, ensure_ascii=False, default=str)
    assert "ACC-A-123456" not in json.dumps(audit.records, ensure_ascii=False, default=str)

    replayed = await runtime.invoke(
        ToolCall("transfer_replay_123456", "transfer", TRANSFER_ARGUMENTS),
        approved_context,
    )
    assert replayed.error_code == "APPROVAL_REQUIRED"
    assert ACCOUNTS[("tenant_a", "ACC-A-123456")] == 99000.0
    assert ACCOUNTS[("tenant_a", "ACC-A-654321")] == 6000.0


@pytest.mark.asyncio
async def test_transfer_timeout_keeps_balances_untouched() -> None:
    runtime, approvals, audit = build_governance()
    context = base_context()
    timeout_arguments = {**TRANSFER_ARGUMENTS, "amount": 90000.0}
    approve_transfer(approvals, "approval_timeout", context, timeout_arguments)
    before = dict(ACCOUNTS)

    result = await runtime.invoke(
        ToolCall("transfer_timeout_999999", "transfer", timeout_arguments),
        replace(context, approval_id="approval_timeout", idempotency_key="transfer_timeout_999999"),
    )

    assert result.ok is False
    assert result.error_code == "TIMEOUT_UNKNOWN"
    assert ACCOUNTS == before
    assert audit.records[-1]["error_code"] == "TIMEOUT_UNKNOWN"
    assert audit.records[-1]["tool_name"] == "transfer"
