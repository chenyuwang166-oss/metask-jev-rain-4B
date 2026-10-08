"""Durable, conservative per-decision slow-path quota.

The lock spans inference: a slow attempt is charged before work starts, while
the denominator grows only after successful work. A crash therefore cannot
create quota credit. One counter file represents one global service history.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import errno
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Iterator


class QuotaStateError(RuntimeError):
    """Counter storage is unreadable or invalid; inference must fail closed."""


@dataclass
class QuotaDecision:
    path: str
    reason: str
    snapshot: dict


class _PathLock:
    def __init__(self) -> None:
        self.mutex = threading.RLock()
        self.local = threading.local()


_REGISTRY_MUTEX = threading.Lock()
_PATH_LOCKS: dict[str, _PathLock] = {}
_ROOT = Path(__file__).resolve().parent


@contextmanager
def _os_lock(path: Path) -> Iterator[None]:
    # Keep the lock file permanently: unlinking it would let another process
    # lock a different inode while an existing process still holds this lock.
    try:
        handle = path.open("x+b")
    except FileExistsError:
        handle = path.open("r+b")
    with handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return  # Windows does not expose a directory fsync via os.open.
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(descriptor)


def cost_settings(config):
    settings = config['routing'].get('cost_fuse', {'enabled': False})
    prices = config['pricing']['models'][config['pricing']['model_key']]
    known = all(isinstance(prices.get(k), (int, float)) and not isinstance(prices[k], bool)
                and math.isfinite(prices[k]) and prices[k] >= 0
                for k in ('input_per_million', 'output_per_million'))
    if settings.get('enabled', True) and not known and config['routing']['mode'] == 'routed':
        raise ValueError('Routed cost fuse requires known profile prices; explicitly use --cost-fuse off')
    return settings, prices, known


class CostFuse:
    def __init__(self, limit_per_1000, *, warmup=100, arm=.85, release=.80, state=None, enabled=True):
        if not math.isfinite(limit_per_1000) or limit_per_1000 <= 0 or not 0 < release < arm <= 1:
            raise ValueError('Invalid cost fuse limits')
        if type(warmup) is not int or warmup < 100: raise ValueError('Cost warmup must be at least 100')
        self.limit = limit_per_1000 / 1000
        self.warmup, self.arm, self.release, self.enabled = warmup, arm, release, enabled
        self.s = dict(n=0, cost=0., n_slow=0, cost_slow=0., engaged=False, transitions=0) if state is None else dict(state)
        v=self.s
        if (any(type(v.get(k)) is not int or v[k]<0 for k in ('n','n_slow','transitions'))
            or v['n_slow']>v['n'] or type(v.get('engaged')) is not bool
            or any(not isinstance(v.get(k),(int,float)) or not math.isfinite(v[k]) or v[k]<0 for k in ('cost','cost_slow'))
            or v['cost_slow']>v['cost']+1e-12): raise QuotaStateError('Invalid cost fuse state')

    def state(self):
        return 'off' if not self.enabled else 'observe' if self.s['n'] < self.warmup else 'engaged' if self.s['engaged'] else 'idle'

    def quota(self, tier_quota):
        s=self.s
        if self.state()!='engaged': return tier_quota,'tier'
        n_fast=s['n']-s['n_slow']
        fast=(s['cost']-s['cost_slow'])/n_fast if n_fast else 0.
        slow=s['cost_slow']/s['n_slow'] if s['n_slow'] else None
        # If even fast exceeds target, remove slow admissions; never alter fast.
        q=0. if fast>=self.release*self.limit else (
            max(0.,(self.release*self.limit-fast)/(slow-fast)) if slow is not None and slow>fast else tier_quota)
        return (q,'cost') if q<tier_quota else (tier_quota,'tier')

    def record(self, cost_usd, slow):
        if not math.isfinite(cost_usd) or cost_usd<0: raise ValueError('Invalid observed cost')
        s=self.s;s['n']+=1;s['cost']+=cost_usd
        if slow:s['n_slow']+=1;s['cost_slow']+=cost_usd
        if not self.enabled or s['n']<self.warmup:return
        running=s['cost']/s['n'];old=s['engaged']
        if not old and running>=self.arm*self.limit:s['engaged']=True
        elif old and running<=self.release*self.limit:s['engaged']=False
        if old!=s['engaged']:s['transitions']+=1

    def snapshot(self, quota):
        q,source=self.quota(quota)
        return dict(state=self.state(),n=self.s['n'],running_cost_per_1000=self.s['cost']/self.s['n']*1000 if self.s['n'] else 0.,
                    effective_quota=q,quota_source=source,transitions=self.s['transitions'],cap_per_1000=self.limit*1000)


def configure_cost(counter, config):
    settings,prices,known=cost_settings(config)
    counter.cost_prices=prices if known else None
    counter.cost_options=dict(limit_per_1000=config['limits']['cost_per_1000_decisions_usd']*(1-config['pricing']['margin']),
        warmup=settings.get('warmup',100),arm=settings.get('arm',.85),release=settings.get('release',.80),
        enabled=settings.get('enabled',True) and known and config['routing']['mode']=='routed')
    counter.cost_fuse=CostFuse(**counter.cost_options)
    counter.tier_quota=config['routing']['quota']


class QuotaCounter:
    """Serialize decisions across threads/processes sharing ``state_path``.

    ``quota`` and ``fuse`` must come from the caller's configuration. Uniform
    experiments can explicitly use quota=fuse=1; an ordinary routed service
    should use its configured limits. Persisted slow attempts include failures.
    """

    def __init__(self, state_path: str | Path, *, quota: float, fuse: float):
        if (
            isinstance(quota, bool)
            or isinstance(fuse, bool)
            or not math.isfinite(quota)
            or not math.isfinite(fuse)
            or not 0 <= quota <= fuse <= 1
        ):
            raise ValueError("Require finite 0 <= quota <= fuse <= 1")
        self.quota = float(quota)
        self.fuse = float(fuse)
        self._quota = Decimal(str(quota))
        self._fuse = Decimal(str(fuse))
        target=Path(state_path)
        if not target.is_absolute():
            raise QuotaStateError('Quota state path must be absolute')
        self.state_path = target.resolve()
        if self.state_path.is_relative_to(_ROOT) or _ROOT.is_relative_to(self.state_path):
            raise QuotaStateError('Quota state must be separate from the package')
        self.lock_path = self.state_path.with_name(self.state_path.name + ".lock")
        if self.lock_path.resolve().parent != self.state_path.parent:
            raise QuotaStateError('Quota lock must stay inside its state directory')
        key = os.path.normcase(str(self.state_path))
        with _REGISTRY_MUTEX:
            self._lock = _PATH_LOCKS.setdefault(key, _PathLock())
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive():
            if self.state_path.exists():
                self._read()
            else:
                self._write({"version": 2, "total": 0, "slow": 0})

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._lock.mutex:
            depth = getattr(self._lock.local, "depth", 0)
            if depth:
                self._lock.local.depth = depth + 1
                try:
                    yield
                finally:
                    self._lock.local.depth = depth
            else:
                with _os_lock(self.lock_path):
                    self._lock.local.depth = 1
                    try:
                        yield
                    finally:
                        self._lock.local.depth = 0

    def _read(self) -> dict:
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, ValueError) as exc:
            raise QuotaStateError("Cannot read quota state; refusing to reset it") from exc
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] not in (1, 2)
            or any(type(state.get(key)) is not int or state[key] < 0 for key in ("total", "slow"))
        ):
            raise QuotaStateError("Invalid quota state; refusing to reset it")
        # slow may exceed total after a failed or interrupted slow attempt.
        return state

    def _write(self, state: dict) -> None:
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.state_path.parent,
                prefix=self.state_path.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary = handle.name
                json.dump(state, handle, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            temporary = None
            _fsync_directory(self.state_path.parent)
        except OSError as exc:
            raise QuotaStateError("Cannot persist quota state; refusing inference") from exc
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _snapshot(state: dict) -> dict:
        total, slow = state["total"], state["slow"]
        return {
            "total": total,
            "slow": slow,
            "slow_fraction": slow / total if total else float(slow > 0),
        }

    def snapshot(self) -> dict:
        with self._exclusive():
            return self._snapshot(self._read())

    def configure_cost(self, config):
        configure_cost(self, config)
        with self._exclusive():self.cost_fuse=CostFuse(**self.cost_options,state=self._read().get('cost_fuse'))

    @contextmanager
    def accounting(self):
        with self._exclusive():
            state=self._read()
            self.cost_fuse=CostFuse(**self.cost_options,state=state.get('cost_fuse'))
            yield

    def record_cost(self, usage, slow):
        with self._exclusive():
            before=self.cost_fuse.snapshot(self.tier_quota)
            if self.cost_prices is not None:
                p=self.cost_prices
                self.cost_fuse.record((usage['input_tokens']*p['input_per_million']+usage['output_tokens']*p['output_per_million'])/1e6,slow)
            after=self.cost_fuse.snapshot(self.tier_quota)
            state=self._read();state.update(version=2,cost_fuse=self.cost_fuse.s);self._write(state)
            if any(before[k]!=after[k] for k in ('state','effective_quota')):
                print('cost_fuse '+json.dumps(after,sort_keys=True),flush=True)
            return after

    @contextmanager
    def decision(self, wants_slow: bool, quota=None) -> Iterator[QuotaDecision]:
        """Reserve, run caller work, then commit only on normal context exit.

        Use ``with counter.decision(wants_slow) as decision`` and run inference
        inside that context. Propagate failures out of the block: swallowing an
        inference failure would incorrectly record a successful decision.
        """
        with self._exclusive():
            if getattr(self._lock.local, "active_decision", False):
                raise RuntimeError("Nested quota decisions are not supported")
            self._lock.local.active_decision = True
            try:
                state = self._read()
                total, slow = state["total"], state["slow"]
                if Decimal(slow) > self._fuse * total:
                    path, reason = "fast", "fuse"
                elif not wants_slow:
                    path, reason = "fast", "router"
                elif Decimal(slow + 1) > self._fuse * (total + 1):
                    path, reason = "fast", "fuse"
                elif Decimal(slow + 1) > min(self._quota, Decimal(str(quota)) if quota is not None else self._quota) * (total + 1):
                    path, reason = "fast", "quota"
                else:
                    path, reason = "slow", "eligible"
                    state["slow"] += 1
                    self._write(state)
                decision = QuotaDecision(path, reason, self._snapshot(state))
                yield decision
                state["total"] += 1
                self._write(state)
                decision.snapshot = self._snapshot(state)
            finally:
                self._lock.local.active_decision = False

class RuntimeCounter:
    """Run-scoped durable counter: startup IO failures abort; later IO failures use fast only."""
    def __init__(self, config):
        import logging
        import uuid
        routing = config['routing']
        self.run_id = routing.get('run_id') or uuid.uuid4().hex
        if not isinstance(self.run_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',self.run_id):
            raise ValueError('run_id must contain 1 to 64 safe filename characters')
        self.memory_only = False
        self._mutex = threading.RLock()
        self._state = {'total':0, 'slow':0}
        self.durable = None
        state_dir=routing.get('state_dir')
        if not state_dir or not Path(state_dir).is_absolute():
            raise QuotaStateError('An absolute state_dir is required')
        state=Path(state_dir).resolve()
        if state.is_relative_to(_ROOT) or _ROOT.is_relative_to(state):
            raise QuotaStateError('State directory must be separate from the package')
        directory=(state/self.run_id).resolve()
        if directory.parent != state:
            raise QuotaStateError('Run directory must stay inside state_dir')
        for name in ('run.json','quota.json','quota.json.lock'):
            if (directory/name).resolve().parent != directory:
                raise QuotaStateError('Run files must stay inside the run directory')
        fingerprint = {k:routing[k] for k in ('mode','quota','fuse')}
        fingerprint.update(model_key=config['model']['model_key'], run_id=self.run_id,
            tier=routing.get('tier'), tiers=routing.get('tiers'), cost_fuse=routing.get('cost_fuse'),
            pricing=config.get('pricing'), cost_cap=config['limits']['cost_per_1000_decisions_usd'],
            basis=config.get('usage',{}).get('basis','engine'))
        try:
            if routing.get('resume'):
                if not routing.get('run_id'):
                    raise ValueError('resume requires an explicit run_id')
                if not (directory/'quota.json').is_file() or not (directory/'run.json').is_file():
                    raise ValueError('Resume state is missing')
                if json.loads((directory/'run.json').read_text(encoding='utf-8')) != fingerprint:
                    raise ValueError('Run fingerprint mismatch')
            else:
                if directory.exists():
                    raise ValueError('run_id already exists; select a new run_id or explicit resume')
                directory.mkdir(parents=True, exist_ok=False)
                with (directory/'run.json').open('x',encoding='utf-8') as handle:
                    json.dump(fingerprint,handle)
            self.durable = QuotaCounter(directory/'quota.json', quota=routing['quota'], fuse=routing['fuse'])
            self._state = self.durable.snapshot()
        except (OSError, QuotaStateError):
            if routing.get('resume'):
                raise ValueError('Resume state cannot be read or validated') from None
            raise QuotaStateError('Quota persistence unavailable; refusing startup') from None

    def configure_cost(self, config):
        configure_cost(self, config)
        if self.durable is not None:
            self.durable.configure_cost(config)
            self.cost_fuse=self.durable.cost_fuse

    @contextmanager
    def accounting(self):
        import logging
        with self._mutex:
            context=None
            if not self.memory_only:
                try:
                    context=self.durable.accounting();context.__enter__()
                    self.cost_fuse=self.durable.cost_fuse
                except (OSError,QuotaStateError):
                    context=None;self.memory_only=True
                    logging.warning('Cost state unavailable; memory counts and fast only')
            try:yield
            except BaseException as exc:
                if context is not None:context.__exit__(type(exc),exc,exc.__traceback__)
                raise
            else:
                if context is not None:context.__exit__(None,None,None)

    def record_cost(self, usage, slow):
        import logging
        before=self.cost_fuse.s['n']
        if not self.memory_only:
            try:return self.durable.record_cost(usage,slow)
            except (OSError,QuotaStateError):
                self.memory_only=True
                logging.warning('Cost commit unavailable; memory counts and fast only')
        # A failed disk commit may already have updated this shared fuse object.
        if self.cost_prices is not None and self.cost_fuse.s['n']==before:
            p=self.cost_prices
            self.cost_fuse.record((usage['input_tokens']*p['input_per_million']+usage['output_tokens']*p['output_per_million'])/1e6,slow)
        return self.cost_fuse.snapshot(self.tier_quota)

    def snapshot(self):
        with self._mutex:
            return dict(self._state)

    @contextmanager
    def decision(self, wants_slow, quota=None):
        import logging
        with self._mutex:
            context = None
            if not self.memory_only:
                try:
                    context = self.durable.decision(wants_slow, quota=quota)
                    decision = context.__enter__()
                    self._state = dict(decision.snapshot)
                except (OSError, QuotaStateError):
                    context = None
                    self.memory_only = True
                    logging.warning('Quota persistence failed; using memory counts and fast only')
            if context is None:
                decision = QuotaDecision('fast', 'memory_only', dict(self._state))
            try:
                yield decision
            except BaseException as exc:
                if context is not None:
                    context.__exit__(type(exc), exc, exc.__traceback__)
                raise
            else:
                self._state['total'] += 1
                if context is not None:
                    try:
                        context.__exit__(None, None, None)
                        self._state = dict(decision.snapshot)
                    except (OSError, QuotaStateError):
                        self.memory_only = True
                        logging.warning('Quota commit failed; using memory counts and fast only')
                decision.snapshot = dict(self._state)
