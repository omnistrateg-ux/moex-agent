"""
MOEX Trading Orchestrator - Multi-LLM Consensus Engine

Coordinates 5 LLM analysts to form trading decisions:
1. OpenAI (GPT-4o) - Structure & Logic
2. Qwen - Alternative Hypotheses
3. Grok - Failure Modes / Stress Test
4. YandexGPT - News Interpreter (STUB)
5. Perplexity - News & Fact Check

Daily target: 5% return
Mode: Aggressive but controlled
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .margin_risk_engine import (
    ContinuationConfig,
    DayMode,
    MarginRiskEngine,
    MarketRegime,
    TierConfig,
    TradeTier,
)

logger = logging.getLogger("moex_agent.orchestrator")


class AnalystDecision(str, Enum):
    """Decision from an analyst."""
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"
    NO_OP = "NO_OP"  # Stub/disabled


class AnalystVerdict(str, Enum):
    """Analyst verdict."""
    SUPPORT = "support"
    CAUTION = "caution"
    REJECT = "reject"
    STUB = "stub"
    UNUSED = "unused"


class OrchestratorDecision(str, Enum):
    """Final orchestrator decision."""
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"
    HALT_DAY = "HALT_DAY"


@dataclass
class AnalystResponse:
    """Response from an LLM analyst."""
    provider: str
    decision: AnalystDecision
    verdict: AnalystVerdict
    ticker: Optional[str] = None
    side: Optional[str] = None
    timeframe: Optional[str] = None
    tier: Optional[TradeTier] = None
    market_regime: Optional[MarketRegime] = None
    confidence: float = 0.0
    expected_r: Optional[float] = None
    expected_pnl_pct: Optional[float] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_prices: List[float] = field(default_factory=list)
    news_risk: str = "unknown"
    reasoning: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    invalidations: List[str] = field(default_factory=list)
    raw_response: Optional[Dict] = None


@dataclass
class TradeProposal:
    """Final trade proposal from orchestrator."""
    timestamp: str
    mode: str  # paper / live
    decision: OrchestratorDecision

    # Daily tracking
    daily_target_pct: float
    daily_target_rub: float
    daily_pnl_rub: float
    day_mode: str

    # Trade details (if TRADE)
    ticker: Optional[str] = None
    side: Optional[str] = None
    setup: Optional[str] = None
    tier: Optional[str] = None
    timeframe: Optional[str] = None

    # Entry
    entry_type: str = "LIMIT"
    entry_price: Optional[float] = None

    # Size
    qty: int = 0
    pct_equity: float = 0.0
    leverage: float = 0.0

    # Risk
    stop_price: Optional[float] = None
    take_prices: List[Dict] = field(default_factory=list)
    max_loss_rub: float = 0.0
    expected_r: float = 0.0

    # Conditions
    invalidations: List[str] = field(default_factory=list)
    news_risk: str = "unknown"
    cost_gate_passed: bool = False
    cost_ratio: float = 0.0

    # Audit
    llm_consensus: Dict[str, str] = field(default_factory=dict)
    agreement_points: List[str] = field(default_factory=list)
    disagreement_points: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)

    # No-trade reason (if NO_TRADE)
    no_trade_reason: Optional[str] = None


class TradingOrchestrator:
    """
    Multi-LLM Trading Orchestrator.

    Coordinates analysts, builds consensus, and makes final trading decisions.
    """

    # Daily limits
    DAILY_TARGET_PCT = 5.0
    MAX_ATTEMPTS_PER_DAY = 3
    LOSS_STREAK_HALT = 2

    def __init__(
        self,
        risk_engine: MarginRiskEngine,
        tier_config: Optional[TierConfig] = None,
        continuation_config: Optional[ContinuationConfig] = None,
        mode: str = "paper",
    ):
        self.risk_engine = risk_engine
        self.tier_config = tier_config or TierConfig()
        self.continuation_config = continuation_config or ContinuationConfig()
        self.mode = mode

        # Analyst clients (to be injected)
        self._openai_client = None
        self._qwen_client = None
        self._grok_client = None
        self._perplexity_client = None

        # Session state
        self.attempts_today = 0
        self.last_responses: Dict[str, AnalystResponse] = {}

        logger.info(f"TradingOrchestrator initialized: mode={mode}")

    def set_clients(
        self,
        openai_client=None,
        qwen_client=None,
        grok_client=None,
        perplexity_client=None,
    ) -> None:
        """Inject LLM client instances."""
        self._openai_client = openai_client
        self._qwen_client = qwen_client
        self._grok_client = grok_client
        self._perplexity_client = perplexity_client

    # ============================================================
    # ANALYST CALLS
    # ============================================================

    def _call_openai(self, state_json: Dict) -> AnalystResponse:
        """Call OpenAI analyst (Structure & Logic)."""
        if self._openai_client is None:
            return AnalystResponse(
                provider="openai",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["OpenAI client not configured"],
            )

        try:
            # TODO: Implement actual OpenAI API call
            # response = self._openai_client.analyze(state_json)
            # return self._parse_openai_response(response)
            return AnalystResponse(
                provider="openai",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["OpenAI integration pending"],
            )
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return AnalystResponse(
                provider="openai",
                decision=AnalystDecision.NO_TRADE,
                verdict=AnalystVerdict.REJECT,
                reasoning=[f"OpenAI error: {str(e)}"],
            )

    def _call_qwen(self, state_json: Dict) -> AnalystResponse:
        """Call Qwen analyst (Alternative Hypotheses)."""
        if self._qwen_client is None:
            return AnalystResponse(
                provider="qwen",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["Qwen client not configured"],
            )

        try:
            # TODO: Implement actual Qwen/Ollama call
            return AnalystResponse(
                provider="qwen",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["Qwen integration pending"],
            )
        except Exception as e:
            logger.error(f"Qwen error: {e}")
            return AnalystResponse(
                provider="qwen",
                decision=AnalystDecision.NO_TRADE,
                verdict=AnalystVerdict.REJECT,
                reasoning=[f"Qwen error: {str(e)}"],
            )

    def _call_grok(self, state_json: Dict) -> AnalystResponse:
        """Call Grok analyst (Failure Modes)."""
        if self._grok_client is None:
            return AnalystResponse(
                provider="grok",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["Grok client not configured"],
            )

        try:
            # TODO: Implement actual Grok API call
            return AnalystResponse(
                provider="grok",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["Grok integration pending"],
            )
        except Exception as e:
            logger.error(f"Grok error: {e}")
            return AnalystResponse(
                provider="grok",
                decision=AnalystDecision.NO_TRADE,
                verdict=AnalystVerdict.REJECT,
                reasoning=[f"Grok error: {str(e)}"],
            )

    def _call_yandexgpt(self, state_json: Dict) -> AnalystResponse:
        """Call YandexGPT (STUB - disabled)."""
        return AnalystResponse(
            provider="yandexgpt",
            decision=AnalystDecision.NO_OP,
            verdict=AnalystVerdict.STUB,
            news_risk="unknown",
            reasoning=["provider_disabled_stub"],
        )

    def _call_perplexity(self, state_json: Dict) -> AnalystResponse:
        """Call Perplexity analyst (News & Fact Check)."""
        if self._perplexity_client is None:
            return AnalystResponse(
                provider="perplexity",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["Perplexity client not configured"],
            )

        try:
            # TODO: Implement actual Perplexity API call
            return AnalystResponse(
                provider="perplexity",
                decision=AnalystDecision.NO_OP,
                verdict=AnalystVerdict.UNUSED,
                reasoning=["Perplexity integration pending"],
            )
        except Exception as e:
            logger.error(f"Perplexity error: {e}")
            return AnalystResponse(
                provider="perplexity",
                decision=AnalystDecision.NO_TRADE,
                verdict=AnalystVerdict.REJECT,
                reasoning=[f"Perplexity error: {str(e)}"],
            )

    # ============================================================
    # CONSENSUS BUILDING
    # ============================================================

    def collect_analyst_responses(self, state_json: Dict) -> Dict[str, AnalystResponse]:
        """Collect responses from all analysts."""
        responses = {
            "openai": self._call_openai(state_json),
            "qwen": self._call_qwen(state_json),
            "grok": self._call_grok(state_json),
            "yandexgpt": self._call_yandexgpt(state_json),
            "perplexity": self._call_perplexity(state_json),
        }

        self.last_responses = responses
        return responses

    def build_consensus(
        self,
        responses: Dict[str, AnalystResponse],
        proposal: Optional[Dict] = None,
    ) -> Tuple[OrchestratorDecision, str, Dict]:
        """
        Build consensus from analyst responses.

        Rules:
        1. Missing critical data → NO_TRADE
        2. Any NO_TRADE by unclear/event risk without strong A+ → NO_TRADE
        3. Ticker/side mismatch → NO_TRADE or size×0.25
        4. Trade only if: tier in {A+,A,B}, has stop, R >= threshold
        5. Never increase risk after loss

        Returns:
            (decision, reason, details)
        """
        details = {
            "agreement_points": [],
            "disagreement_points": [],
            "red_flags": [],
            "reasoning": [],
        }

        # Count active responses (not STUB or UNUSED)
        active_responses = [
            r for r in responses.values()
            if r.verdict not in (AnalystVerdict.STUB, AnalystVerdict.UNUSED)
        ]

        # If no active analysts, NO_TRADE
        if not active_responses:
            return (
                OrchestratorDecision.NO_TRADE,
                "No active analysts available",
                details,
            )

        # Count verdicts
        support_count = sum(1 for r in active_responses if r.verdict == AnalystVerdict.SUPPORT)
        caution_count = sum(1 for r in active_responses if r.verdict == AnalystVerdict.CAUTION)
        reject_count = sum(1 for r in active_responses if r.verdict == AnalystVerdict.REJECT)

        # Collect red flags from Grok
        grok_response = responses.get("grok")
        if grok_response and grok_response.red_flags:
            details["red_flags"] = grok_response.red_flags

        # Rule: Any reject without strong support → NO_TRADE
        if reject_count > 0 and support_count < 2:
            return (
                OrchestratorDecision.NO_TRADE,
                f"Analyst rejection: {reject_count} reject vs {support_count} support",
                details,
            )

        # Rule: Multiple red flags → NO_TRADE
        if len(details["red_flags"]) >= 3:
            return (
                OrchestratorDecision.NO_TRADE,
                f"Too many red flags: {len(details['red_flags'])}",
                details,
            )

        # Rule: Check news risk
        news_risk = "unknown"
        perplexity_response = responses.get("perplexity")
        if perplexity_response and perplexity_response.news_risk != "unknown":
            news_risk = perplexity_response.news_risk

        if news_risk == "high":
            return (
                OrchestratorDecision.NO_TRADE,
                "High news risk",
                details,
            )

        # Check ticker/side agreement
        tickers = set()
        sides = set()
        for r in active_responses:
            if r.ticker:
                tickers.add(r.ticker)
            if r.side:
                sides.add(r.side)

        if len(tickers) > 1:
            details["disagreement_points"].append(f"Ticker mismatch: {tickers}")
            return (
                OrchestratorDecision.NO_TRADE,
                f"Ticker disagreement: {tickers}",
                details,
            )

        if len(sides) > 1:
            details["disagreement_points"].append(f"Side mismatch: {sides}")
            return (
                OrchestratorDecision.NO_TRADE,
                f"Side disagreement: {sides}",
                details,
            )

        # Majority support → TRADE
        if support_count >= 2:
            details["reasoning"].append(f"Consensus: {support_count} support, {caution_count} caution")
            return (
                OrchestratorDecision.TRADE,
                f"Consensus reached: {support_count} support",
                details,
            )

        # Majority caution with no reject → TRADE with reduced size
        if caution_count >= 2 and reject_count == 0:
            details["reasoning"].append("Caution consensus - consider reduced size")
            return (
                OrchestratorDecision.TRADE,
                "Caution consensus - reduced size recommended",
                details,
            )

        # Default: NO_TRADE
        return (
            OrchestratorDecision.NO_TRADE,
            "No clear consensus",
            details,
        )

    # ============================================================
    # MAIN ORCHESTRATION
    # ============================================================

    def check_pre_conditions(self) -> Tuple[bool, str]:
        """
        Check pre-conditions before orchestration.

        Returns:
            (can_proceed, reason)
        """
        # Check kill switch
        kill_active, kill_reason = self.risk_engine.check_kill_switch()
        if kill_active:
            return False, f"Kill switch active: {kill_reason}"

        # Check day mode
        day_mode = self.risk_engine.update_day_mode(self.continuation_config)
        if day_mode == DayMode.HALT:
            return False, "Day mode is HALT"

        # Check attempts
        if self.attempts_today >= self.MAX_ATTEMPTS_PER_DAY:
            # Allow 4th attempt only if first trade was win and we're in trend
            if self.attempts_today == self.MAX_ATTEMPTS_PER_DAY:
                if self.risk_engine.state.first_trade_win:
                    logger.info("Allowing 4th attempt (first trade was win)")
                else:
                    return False, f"Max attempts reached: {self.attempts_today}"

        # Check loss streak
        if self.risk_engine.state.consecutive_losses >= self.LOSS_STREAK_HALT:
            return False, f"Loss streak halt: {self.risk_engine.state.consecutive_losses} losses"

        return True, "Pre-conditions OK"

    def orchestrate(
        self,
        state_json: Dict,
    ) -> TradeProposal:
        """
        Main orchestration method.

        1. Check pre-conditions
        2. Collect analyst responses
        3. Build consensus
        4. Validate with risk engine
        5. Return trade proposal or NO_TRADE

        Args:
            state_json: Current market state with quotes, candles, portfolio, etc.

        Returns:
            TradeProposal with decision and details
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        equity = self.risk_engine.state.equity
        daily_target_rub = equity * (self.DAILY_TARGET_PCT / 100)

        # Base proposal
        proposal = TradeProposal(
            timestamp=timestamp,
            mode=self.mode,
            decision=OrchestratorDecision.NO_TRADE,
            daily_target_pct=self.DAILY_TARGET_PCT,
            daily_target_rub=daily_target_rub,
            daily_pnl_rub=self.risk_engine.state.daily_pnl,
            day_mode=self.risk_engine.state.day_mode.value,
        )

        # 1. Check pre-conditions
        can_proceed, reason = self.check_pre_conditions()
        if not can_proceed:
            proposal.decision = OrchestratorDecision.HALT_DAY
            proposal.no_trade_reason = reason
            logger.warning(f"Pre-condition failed: {reason}")
            return proposal

        # 2. Collect analyst responses
        responses = self.collect_analyst_responses(state_json)

        # Build LLM consensus dict for audit
        proposal.llm_consensus = {
            provider: r.verdict.value
            for provider, r in responses.items()
        }

        # 3. Build consensus
        decision, consensus_reason, details = self.build_consensus(responses)
        proposal.agreement_points = details.get("agreement_points", [])
        proposal.disagreement_points = details.get("disagreement_points", [])
        proposal.reasoning = details.get("reasoning", [])

        if decision == OrchestratorDecision.NO_TRADE:
            proposal.decision = OrchestratorDecision.NO_TRADE
            proposal.no_trade_reason = consensus_reason
            return proposal

        if decision == OrchestratorDecision.HALT_DAY:
            proposal.decision = OrchestratorDecision.HALT_DAY
            proposal.no_trade_reason = consensus_reason
            return proposal

        # 4. Extract trade details from consensus
        # Use the first active response with trade details
        trade_response = None
        for r in responses.values():
            if r.decision in (AnalystDecision.LONG, AnalystDecision.SHORT):
                trade_response = r
                break

        if not trade_response:
            proposal.decision = OrchestratorDecision.NO_TRADE
            proposal.no_trade_reason = "No trade details in responses"
            return proposal

        # 5. Validate tier and cost gate
        tier = trade_response.tier or TradeTier.C
        expected_r = trade_response.expected_r or 0.0
        expected_pnl_pct = trade_response.expected_pnl_pct or 0.0

        if tier == TradeTier.C:
            proposal.decision = OrchestratorDecision.NO_TRADE
            proposal.no_trade_reason = f"Tier C - NO TRADE (R={expected_r:.1f})"
            return proposal

        # Check continuation mode constraints
        if self.risk_engine.state.day_mode == DayMode.CONTINUATION:
            allowed, reason = self.risk_engine.check_continuation_allowed(
                tier=tier,
                expected_r=expected_r,
                regime=trade_response.market_regime or MarketRegime.UNKNOWN,
                continuation_config=self.continuation_config,
            )
            if not allowed:
                proposal.decision = OrchestratorDecision.NO_TRADE
                proposal.no_trade_reason = f"Continuation not allowed: {reason}"
                return proposal

        # 6. Cost gate check
        spread_pct = state_json.get("quotes", {}).get(
            trade_response.ticker, {}
        ).get("spread_pct", 0.15)

        entry_price = trade_response.entry_price or 0
        take_price = trade_response.take_prices[0] if trade_response.take_prices else 0
        expected_gain_pct = abs(take_price - entry_price) / entry_price * 100 if entry_price else 0

        cost_passed, cost_ratio, cost_reason = self.risk_engine.check_cost_gate(
            spread_pct=spread_pct,
            expected_gain_pct=expected_gain_pct,
            tier_config=self.tier_config,
        )

        proposal.cost_gate_passed = cost_passed
        proposal.cost_ratio = cost_ratio

        if not cost_passed:
            proposal.decision = OrchestratorDecision.NO_TRADE
            proposal.no_trade_reason = cost_reason
            return proposal

        # 7. Build final proposal
        risk_pct = self.risk_engine.get_risk_for_tier(
            tier=tier,
            tier_config=self.tier_config,
            continuation_config=self.continuation_config,
        )

        proposal.decision = OrchestratorDecision.TRADE
        proposal.ticker = trade_response.ticker
        proposal.side = trade_response.side or trade_response.decision.value
        proposal.setup = "ml_signal"  # TODO: get from response
        proposal.tier = tier.value
        proposal.timeframe = trade_response.timeframe
        proposal.entry_price = trade_response.entry_price
        proposal.stop_price = trade_response.stop_price
        proposal.take_prices = [{"price": p, "pct": 50} for p in trade_response.take_prices[:2]]
        proposal.expected_r = expected_r
        proposal.pct_equity = risk_pct
        proposal.news_risk = trade_response.news_risk
        proposal.invalidations = trade_response.invalidations

        # Calculate position size
        if proposal.entry_price and proposal.stop_price:
            stop_distance = abs(proposal.entry_price - proposal.stop_price)
            max_loss = equity * (risk_pct / 100)
            if stop_distance > 0:
                proposal.qty = int(max_loss / stop_distance)
                proposal.max_loss_rub = max_loss
                proposal.leverage = (proposal.qty * proposal.entry_price) / equity

        self.attempts_today += 1
        logger.info(f"Trade proposal: {proposal.ticker} {proposal.side} tier={proposal.tier}")

        return proposal

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.attempts_today = 0
        self.last_responses = {}
        logger.info("Orchestrator daily counters reset")

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "mode": self.mode,
            "attempts_today": self.attempts_today,
            "max_attempts": self.MAX_ATTEMPTS_PER_DAY,
            "daily_target_pct": self.DAILY_TARGET_PCT,
            "day_mode": self.risk_engine.state.day_mode.value,
            "loss_streak": self.risk_engine.state.consecutive_losses,
            "loss_streak_halt": self.LOSS_STREAK_HALT,
            "last_analysts": list(self.last_responses.keys()),
        }


def create_orchestrator(
    initial_equity: float = 200000,
    mode: str = "paper",
) -> TradingOrchestrator:
    """Factory function to create orchestrator with default config."""
    risk_engine = MarginRiskEngine(initial_equity=initial_equity)
    return TradingOrchestrator(
        risk_engine=risk_engine,
        mode=mode,
    )
