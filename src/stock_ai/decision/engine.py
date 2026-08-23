"""Daily whole-portfolio comparison against the unchanged HOLD counterfactual."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stock_ai.decision.costs import TransactionCostEngine
from stock_ai.decision.tax import SaleTaxInput, SimpleJapanTaxEngine, TaxEstimate
from stock_ai.domain import (
    AccountBucket,
    AccountType,
    PortfolioProposal,
    PortfolioState,
    Position,
    Prediction,
    ProposalAction,
    ProposalLine,
    Security,
    TargetPosition,
    TransactionCostEstimate,
    WithholdingMode,
)


class SearchSpaceTooLarge(RuntimeError):
    """Fail-closed signal rather than silently using a different optimizer."""


class DecisionCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    security: Security
    account_bucket_id: str = Field(min_length=1)
    price: Decimal = Field(gt=0)
    average_daily_trading_value: Decimal | None = Field(default=None, gt=0)
    prediction: Prediction

    @model_validator(mode="after")
    def prediction_matches_security(self) -> DecisionCandidate:
        if self.prediction.symbol != self.security.symbol:
            raise ValueError("prediction symbol must match candidate security")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.security.symbol, self.account_bucket_id)


class DecisionEngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "decision-engine-v1"
    lot_size: int = Field(default=100, gt=0)
    maximum_positions: int = Field(default=10, gt=0)
    maximum_symbol_weight: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    maximum_sector_weight: Decimal = Field(default=Decimal("0.30"), gt=0, le=1)
    minimum_cash_ratio: Decimal = Field(default=Decimal("0.10"), ge=0, le=1)
    maximum_turnover_ratio: Decimal = Field(default=Decimal("1.0"), gt=0)
    minimum_improvement_yen: Decimal = Field(default=Decimal("1000"), ge=0)
    uncertainty_buffer_yen: Decimal = Field(default=Decimal("0"), ge=0)
    implementation_buffer_yen: Decimal = Field(default=Decimal("0"), ge=0)
    downside_penalty_weight: Decimal = Field(default=Decimal("1"), ge=0)
    uncertainty_penalty_weight: Decimal = Field(default=Decimal("1"), ge=0)
    large_loss_probability_penalty_weight: Decimal = Field(default=Decimal("0"), ge=0)
    concentration_penalty_weight: Decimal = Field(default=Decimal("0"), ge=0)
    turnover_penalty_bps: Decimal = Field(default=Decimal("0"), ge=0)
    maximum_trade_adv_ratio: Decimal = Field(default=Decimal("0.05"), gt=0, le=1)
    horizon_weights: tuple[Decimal, Decimal, Decimal] = (
        Decimal("0.20"),
        Decimal("0.50"),
        Decimal("0.30"),
    )
    max_search_combinations: int = Field(default=2_000_000, gt=0)

    @field_validator("horizon_weights")
    @classmethod
    def horizon_weights_sum_to_one(
        cls, value: tuple[Decimal, Decimal, Decimal]
    ) -> tuple[Decimal, Decimal, Decimal]:
        if sum(value) != Decimal("1"):
            raise ValueError("horizon weights must sum to one")
        return value


@dataclass(frozen=True)
class _Evaluation:
    targets: dict[tuple[str, str], int]
    utility: Decimal
    gross_signal_utility: Decimal
    cash_after: dict[str, Decimal]
    costs: dict[tuple[str, str], TransactionCostEstimate]
    taxes: dict[tuple[str, str], TaxEstimate]
    turnover: Decimal


def classify_action(current_shares: int, target_shares: int, band: int = 0) -> ProposalAction:
    if min(current_shares, target_shares, band) < 0:
        raise ValueError("shares and band must be non-negative")
    if current_shares == 0 and target_shares == 0:
        return ProposalAction.SKIP
    if current_shares > 0 and target_shares == 0:
        return ProposalAction.SELL
    if target_shares > current_shares + band:
        return ProposalAction.BUY
    if abs(target_shares - current_shares) <= band:
        return ProposalAction.HOLD
    return ProposalAction.REDUCE


class DailyPortfolioDecisionEngine:
    """Exact discrete search for a bounded candidate universe.

    The explicit search cap is intentional.  This Goal does not silently swap in
    an unreviewed heuristic when the configured production universe is too large.
    """

    def __init__(
        self,
        *,
        config: DecisionEngineConfig,
        cost_engine: TransactionCostEngine,
        tax_engine: SimpleJapanTaxEngine,
    ) -> None:
        self.config = config
        self.cost_engine = cost_engine
        self.tax_engine = tax_engine

    def propose(
        self,
        *,
        portfolio: PortfolioState,
        candidates: tuple[DecisionCandidate, ...],
        generated_at: datetime,
        model_bundle_version: str,
    ) -> PortfolioProposal:
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if self.tax_engine.policy.effective_from > portfolio.as_of.date():
            raise ValueError("configured tax policy is not effective at the portfolio as_of")
        ordered_candidates = tuple(sorted(candidates, key=lambda candidate: candidate.key))
        candidate_map = {candidate.key: candidate for candidate in ordered_candidates}
        if len(candidate_map) != len(candidates):
            raise ValueError("decision candidates must be unique by symbol/account bucket")
        current = portfolio.position_map()
        missing_holdings = set(current) - set(candidate_map)
        if missing_holdings:
            raise ValueError(f"all current holdings must be evaluated: {sorted(missing_holdings)}")
        bucket_map = {bucket.bucket_id: bucket for bucket in portfolio.account_buckets}
        if unknown := {candidate.account_bucket_id for candidate in candidates} - set(bucket_map):
            raise ValueError(f"candidates reference unknown account buckets: {sorted(unknown)}")
        for candidate in candidates:
            bucket = bucket_map[candidate.account_bucket_id]
            if bucket.fee_policy_id != self.cost_engine.policy.policy_id:
                raise ValueError(
                    "account bucket fee policy does not match the configured cost engine"
                )
            if bucket.tax_policy_id != self.tax_engine.policy.policy_id:
                raise ValueError(
                    "account bucket tax policy does not match the configured tax engine"
                )
            if candidate.security.lot_size != self.config.lot_size:
                raise ValueError("candidate lot size does not match the decision-engine lot size")
            position = current.get(candidate.key)
            if position is not None and position.market_price != candidate.price:
                raise ValueError("current position and candidate reference prices must match")
            if position is not None and (
                bucket.account_type is AccountType.UNKNOWN_MANUAL
                or bucket.withholding_mode is WithholdingMode.UNKNOWN
            ):
                unrealized = (
                    candidate.price - position.average_acquisition_price
                ) * position.shares
                if unrealized != 0:
                    raise ValueError("material tax impact in an unknown account must fail closed")
        symbols: dict[str, tuple[Decimal, str, int]] = {}
        for candidate in candidates:
            metadata = (candidate.price, candidate.security.sector, candidate.security.lot_size)
            previous = symbols.setdefault(candidate.security.symbol, metadata)
            if previous != metadata:
                raise ValueError(
                    "same-symbol candidates must use identical price, sector, and lot size"
                )
        if any(candidate.prediction.as_of > portfolio.as_of for candidate in candidates):
            raise ValueError("predictions newer than the portfolio as_of cannot be used")
        prediction_provenance = {
            (
                candidate.prediction.model_version,
                candidate.prediction.feature_version,
                candidate.prediction.data_snapshot_id,
            )
            for candidate in candidates
        }
        if len(prediction_provenance) > 1:
            raise ValueError("all candidates must use one coherent prediction bundle")
        prediction_model_version = (
            next(iter(prediction_provenance))[0] if prediction_provenance else model_bundle_version
        )
        if prediction_provenance and prediction_model_version != model_bundle_version:
            raise ValueError("declared model bundle version does not match candidate predictions")
        if any(state.tax_year != portfolio.as_of.year for state in portfolio.tax_states):
            raise ValueError("tax state year must match the portfolio as_of year")

        total_wealth = sum((item.deployable_cash for item in portfolio.cash), Decimal("0"))
        total_wealth += sum(
            (position.market_value for position in portfolio.positions), Decimal("0")
        )
        current_targets = {key: position.shares for key, position in current.items()}
        current_targets.update({key: 0 for key in candidate_map if key not in current_targets})
        hold = self._evaluate(
            targets=current_targets,
            portfolio=portfolio,
            candidates=candidate_map,
            buckets=bucket_map,
            total_wealth=total_wealth,
            enforce_trade_constraints=False,
        )
        if hold is None:
            raise RuntimeError("the unchanged HOLD scenario must always be evaluable")

        option_sets = [
            self._share_options(candidate, current.get(key), total_wealth)
            for key, candidate in candidate_map.items()
        ]
        combination_count = math.prod(len(options) for options in option_sets)
        if combination_count > self.config.max_search_combinations:
            raise SearchSpaceTooLarge(
                f"{combination_count:,} target combinations exceed the configured safe cap "
                f"of {self.config.max_search_combinations:,}"
            )
        keys = tuple(candidate_map)
        best = hold
        for values in itertools.product(*option_sets):
            targets = dict(zip(keys, values, strict=True))
            evaluation = self._evaluate(
                targets=targets,
                portfolio=portfolio,
                candidates=candidate_map,
                buckets=bucket_map,
                total_wealth=total_wealth,
                enforce_trade_constraints=True,
            )
            if evaluation is not None and evaluation.utility > best.utility:
                best = evaluation

        threshold = (
            self.config.minimum_improvement_yen
            + self.config.uncertainty_buffer_yen
            + self.config.implementation_buffer_yen
        )
        raw_improvement = best.utility - hold.utility
        no_trade_reason: str | None = None
        if raw_improvement <= threshold:
            best = hold
            no_trade_reason = (
                f"best net improvement {raw_improvement:.2f} JPY did not exceed "
                f"the no-trade threshold {threshold:.2f} JPY"
            )
        return self._proposal(
            portfolio=portfolio,
            candidates=candidate_map,
            hold=hold,
            selected=best,
            generated_at=generated_at,
            model_bundle_version=model_bundle_version,
            no_trade_reason=no_trade_reason,
        )

    def _share_options(
        self, candidate: DecisionCandidate, position: Position | None, total_wealth: Decimal
    ) -> tuple[int, ...]:
        current_shares = position.shares if position else 0
        symbol_cap = total_wealth * self.config.maximum_symbol_weight
        capped_lots = int(symbol_cap // (candidate.price * self.config.lot_size))
        maximum_shares = max(current_shares, capped_lots * self.config.lot_size)
        remainder = current_shares % self.config.lot_size
        first_target = remainder if remainder else 0
        options = set(range(first_target, maximum_shares + 1, self.config.lot_size))
        if remainder == 0:
            options.add(0)
        options.add(current_shares)
        return tuple(sorted(options))

    def _signal_utility(self, candidate: DecisionCandidate, shares: int) -> Decimal:
        prediction = candidate.prediction
        weights = self.config.horizon_weights
        expected_return = (
            weights[0] * Decimal(str(prediction.expected_return_1d))
            + weights[1] * Decimal(str(prediction.expected_return_5d))
            + weights[2] * Decimal(str(prediction.expected_return_20d))
        )
        downside = Decimal(str(max(0.0, -prediction.downside_quantile)))
        uncertainty = Decimal(str(prediction.uncertainty.combined))
        large_loss_probability = Decimal(str(prediction.large_loss_probability))
        market_value = candidate.price * shares
        return market_value * (
            expected_return
            - self.config.downside_penalty_weight * downside
            - self.config.uncertainty_penalty_weight * uncertainty
            - self.config.large_loss_probability_penalty_weight * large_loss_probability
        )

    def _evaluate(
        self,
        *,
        targets: dict[tuple[str, str], int],
        portfolio: PortfolioState,
        candidates: dict[tuple[str, str], DecisionCandidate],
        buckets: dict[str, AccountBucket],
        total_wealth: Decimal,
        enforce_trade_constraints: bool,
    ) -> _Evaluation | None:
        current = portfolio.position_map()
        cash_after = {key: item.deployable_cash for key, item in portfolio.cash_map().items()}
        costs: dict[tuple[str, str], TransactionCostEstimate] = {}
        taxes: dict[tuple[str, str], TaxEstimate] = {}
        sales_by_bucket: dict[str, list[SaleTaxInput]] = {}
        turnover = Decimal("0")
        gross_signal_utility = Decimal("0")
        target_symbol_values: dict[str, Decimal] = {}
        current_symbol_values: dict[str, Decimal] = {}
        target_sector_values: dict[str, Decimal] = {}
        current_sector_values: dict[str, Decimal] = {}
        target_symbols: set[str] = set()
        current_symbols = {position.symbol for position in current.values() if position.shares > 0}
        tax_states = portfolio.tax_state_map()

        for key, candidate in candidates.items():
            position = current.get(key)
            current_shares = position.shares if position else 0
            target_shares = targets[key]
            if target_shares < 0:
                return None
            difference = target_shares - current_shares
            gross_signal_utility += self._signal_utility(candidate, target_shares)
            target_value = candidate.price * target_shares
            current_value = candidate.price * current_shares
            target_symbol_values[candidate.security.symbol] = (
                target_symbol_values.get(candidate.security.symbol, Decimal("0")) + target_value
            )
            current_symbol_values[candidate.security.symbol] = (
                current_symbol_values.get(candidate.security.symbol, Decimal("0")) + current_value
            )
            target_sector_values[candidate.security.sector] = (
                target_sector_values.get(candidate.security.sector, Decimal("0")) + target_value
            )
            current_sector_values[candidate.security.sector] = (
                current_sector_values.get(candidate.security.sector, Decimal("0")) + current_value
            )
            if target_shares > 0:
                target_symbols.add(candidate.security.symbol)
            if difference == 0:
                costs[key] = TransactionCostEstimate(policy_version=self.cost_engine.policy.version)
                continue
            trade_shares = abs(difference)
            trade_cost = self.cost_engine.estimate(
                shares=trade_shares,
                price=candidate.price,
                average_daily_trading_value=candidate.average_daily_trading_value,
            )
            costs[key] = trade_cost
            notional = candidate.price * trade_shares
            turnover += notional
            if enforce_trade_constraints:
                if candidate.average_daily_trading_value is None:
                    return None
                if (
                    notional
                    > candidate.average_daily_trading_value * self.config.maximum_trade_adv_ratio
                ):
                    return None
            if difference > 0:
                cash_after[candidate.account_bucket_id] -= notional + trade_cost.total
            else:
                if position is None:
                    return None
                sales_by_bucket.setdefault(candidate.account_bucket_id, []).append(
                    SaleTaxInput(
                        allocation_id=f"{key[0]}:{key[1]}",
                        position=position,
                        sell_shares=trade_shares,
                        expected_sell_price=candidate.price,
                        estimated_deductible_cost=trade_cost.commission,
                    )
                )
                cash_after[candidate.account_bucket_id] += notional - trade_cost.total

        for bucket_id in sorted(sales_by_bucket):
            allocated = self.tax_engine.estimate_sales(
                account_bucket=buckets[bucket_id],
                sales=tuple(sales_by_bucket[bucket_id]),
                tax_state=tax_states[bucket_id],
            )
            for allocation_id, tax in allocated.items():
                symbol, allocation_bucket = allocation_id.split(":", maxsplit=1)
                taxes[(symbol, allocation_bucket)] = tax
                cash_after[bucket_id] -= tax.estimated_cash_withholding

        if any(value < 0 for value in cash_after.values()):
            return None
        if enforce_trade_constraints:
            if len(target_symbols) > self.config.maximum_positions:
                moves_toward_limit = target_symbols < current_symbols
                if not moves_toward_limit:
                    return None
            symbol_cap = total_wealth * self.config.maximum_symbol_weight
            for symbol, target_value in target_symbol_values.items():
                if target_value > symbol_cap and target_value > current_symbol_values.get(
                    symbol, Decimal("0")
                ):
                    return None
            sector_cap = total_wealth * self.config.maximum_sector_weight
            for sector, target_value in target_sector_values.items():
                if target_value > sector_cap and target_value > current_sector_values.get(
                    sector, Decimal("0")
                ):
                    return None
            minimum_cash = total_wealth * self.config.minimum_cash_ratio
            target_cash = sum(cash_after.values(), Decimal("0"))
            if target_cash < minimum_cash:
                return None
            if turnover > total_wealth * self.config.maximum_turnover_ratio:
                return None

        if total_wealth > 0:
            concentration_penalty = self.config.concentration_penalty_weight * sum(
                (value * value / total_wealth for value in target_symbol_values.values()),
                Decimal("0"),
            )
        else:
            concentration_penalty = Decimal("0")
        gross_signal_utility -= concentration_penalty
        total_cost = sum((item.total for item in costs.values()), Decimal("0"))
        total_tax = sum(
            (item.immediate_tax_effect + item.nisa_opportunity_cost for item in taxes.values()),
            Decimal("0"),
        )
        turnover_penalty = turnover * self.config.turnover_penalty_bps / Decimal("10000")
        utility = gross_signal_utility - total_cost - total_tax - turnover_penalty
        return _Evaluation(
            targets=targets,
            utility=utility,
            gross_signal_utility=gross_signal_utility,
            cash_after=cash_after,
            costs=costs,
            taxes=taxes,
            turnover=turnover,
        )

    def _proposal(
        self,
        *,
        portfolio: PortfolioState,
        candidates: dict[tuple[str, str], DecisionCandidate],
        hold: _Evaluation,
        selected: _Evaluation,
        generated_at: datetime,
        model_bundle_version: str,
        no_trade_reason: str | None,
    ) -> PortfolioProposal:
        current = portfolio.position_map()
        lines: list[ProposalLine] = []
        targets: list[TargetPosition] = []
        for key, candidate in candidates.items():
            position = current.get(key)
            current_shares = position.shares if position else 0
            target_shares = selected.targets[key]
            difference = target_shares - current_shares
            cost = selected.costs.get(
                key, TransactionCostEstimate(policy_version=self.cost_engine.policy.version)
            )
            tax = selected.taxes.get(key)
            tax_effect = tax.immediate_tax_effect if tax else Decimal("0")
            nisa_opportunity_cost = tax.nisa_opportunity_cost if tax else Decimal("0")
            current_signal = self._signal_utility(candidate, current_shares)
            target_signal = self._signal_utility(candidate, target_shares)
            notional = candidate.price * abs(difference)
            line_improvement = (
                target_signal - current_signal - cost.total - tax_effect - nisa_opportunity_cost
            )
            line_improvement -= notional * self.config.turnover_penalty_bps / Decimal("10000")
            cash_required_or_released = (
                notional + cost.total
                if difference > 0
                else -(
                    notional
                    - cost.total
                    - (tax.estimated_cash_withholding if tax else Decimal("0"))
                )
            )
            action = classify_action(current_shares, target_shares)
            reasons = [f"ACTION_{action.value}", "WHOLE_PORTFOLIO_HOLD_COMPARISON"]
            human_reasons = ["現在ポートフォリオを維持する場合との全体比較を実施"]
            if action in {ProposalAction.HOLD, ProposalAction.SKIP} and no_trade_reason:
                reasons.append("NO_TRADE_ZONE")
                human_reasons.append("純改善がノートレード基準を超えないため株数変更なし")
            elif action is ProposalAction.HOLD:
                human_reasons.append("全体最適化後もこの口座バケットの保有株数を維持")
            elif action is ProposalAction.SKIP:
                human_reasons.append("現金・リスク・税・コスト考慮後の目標株数は0株")
            else:
                human_reasons.append("税・取引コスト控除後の目標株数へ変更")
            lines.append(
                ProposalLine(
                    line_id=f"{candidate.security.symbol}:{candidate.account_bucket_id}",
                    symbol=candidate.security.symbol,
                    company_name=candidate.security.company_name,
                    account_bucket_id=candidate.account_bucket_id,
                    current_shares=current_shares,
                    recommended_shares=target_shares,
                    share_difference=difference,
                    action=action,
                    reference_price=candidate.price,
                    current_market_value=candidate.price * current_shares,
                    recommended_market_value=candidate.price * target_shares,
                    estimated_cash_required_or_released=cash_required_or_released,
                    hold_expected_value=current_signal,
                    proposed_expected_value=target_signal,
                    transaction_cost=cost,
                    estimated_tax_effect=tax_effect,
                    estimated_nisa_opportunity_cost=nisa_opportunity_cost,
                    estimated_realized_pnl=(tax.realized_pnl if tax else Decimal("0")),
                    estimated_tax_cash_withholding=(
                        tax.estimated_cash_withholding if tax else Decimal("0")
                    ),
                    tax_policy_version=self.tax_engine.policy.version,
                    tax_is_estimate=(tax.is_estimate if tax else True),
                    tax_assumptions=(tax.assumptions if tax else ("no sale",)),
                    net_expected_improvement=line_improvement,
                    downside_risk=candidate.prediction.downside_quantile,
                    uncertainty=candidate.prediction.uncertainty.combined,
                    reason_codes=tuple(reasons),
                    human_readable_reasons=tuple(human_reasons),
                )
            )
            targets.append(
                TargetPosition(
                    symbol=candidate.security.symbol,
                    account_bucket_id=candidate.account_bucket_id,
                    target_shares=target_shares,
                )
            )
        targets.sort(key=lambda target: target.key)
        lines.sort(key=lambda line: (line.symbol, line.account_bucket_id))
        identity_parts = [
            portfolio.portfolio_id,
            portfolio.as_of.isoformat(),
            generated_at.isoformat(),
            model_bundle_version,
            self.config.model_dump_json(),
            self.cost_engine.policy.model_dump_json(),
            self.tax_engine.policy.model_dump_json(),
            *(
                f"{target.symbol}:{target.account_bucket_id}:{target.target_shares}"
                for target in targets
            ),
        ]
        identifier_source = "|".join(identity_parts)
        identifier = hashlib.sha256(identifier_source.encode()).hexdigest()[:16]
        return PortfolioProposal(
            proposal_id=f"proposal-{portfolio.as_of.date()}-{identifier}",
            as_of=portfolio.as_of,
            generated_at=generated_at,
            current_portfolio_id=portfolio.portfolio_id,
            targets=tuple(targets),
            lines=tuple(lines),
            hold_utility=hold.utility,
            proposed_utility=selected.utility,
            net_improvement=selected.utility - hold.utility,
            estimated_cash_after=selected.cash_after,
            model_bundle_version=model_bundle_version,
            decision_engine_version=self.config.version,
            cost_policy_id=self.cost_engine.policy.policy_id,
            cost_policy_version=self.cost_engine.policy.version,
            tax_policy_id=self.tax_engine.policy.policy_id,
            tax_policy_version=self.tax_engine.policy.version,
            no_trade_reason=no_trade_reason,
        )
