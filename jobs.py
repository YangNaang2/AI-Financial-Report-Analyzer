"""Bounded, owner-scoped background work; reruns only observe existing jobs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock, Event
from datetime import datetime, timezone
from uuid import uuid4

class JobError(ValueError):
    pass

class JobCancelled(JobError):
    pass

@dataclass
class Context:
    update: object
    event: Event
    def progress(self, fraction, message):
        self.check_cancelled()
        self.update(max(0., min(1., float(fraction))), str(message))
    def check_cancelled(self):
        if self.event.is_set():
            raise JobCancelled('사용자가 작업을 중지했습니다.')

class JobManager:
    def __init__(self, max_workers=2):
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='report-lens')
        self.lock = RLock()
        self.jobs = {}
    def submit(self, owner, kind, fn):
        if not owner:
            raise JobError('작업 공간이 필요합니다.')
        with self.lock:
            if any(j['owner'] == owner and j['state'] in ('queued','running') for j in self.jobs.values()):
                raise JobError('진행 중인 작업을 먼저 완료하거나 중지하세요.')
            # Retain a bounded history without evicting active jobs.
            done = [k for k,v in self.jobs.items() if v['state'] not in ('queued','running')]
            for k in done[:-40]:
                del self.jobs[k]
            ident = uuid4().hex
            self.jobs[ident] = dict(id=ident, owner=owner, kind=kind, state='queued', progress=0., message='작업 대기', result=None, error=None, created_at=datetime.now(timezone.utc).isoformat(), event=Event())
        self.pool.submit(self._run, ident, fn)
        return ident
    def _run(self, ident, fn):
        def update(fraction, message):
            with self.lock:
                self.jobs[ident].update(progress=fraction, message=message)
        with self.lock:
            job = self.jobs[ident]
            job['state'] = 'running'
            ctx = Context(update, job['event'])
        try:
            ctx.check_cancelled()
            result = fn(ctx)
            with self.lock:
                job.update(state='completed', progress=1., message='처리 완료', result=result)
        except JobCancelled as exc:
            with self.lock:
                job.update(state='cancelled', error=str(exc), message='중지됨')
        except Exception as exc:
            with self.lock:
                job.update(state='failed', error=f'{type(exc).__name__}: {exc}', message='처리 실패')
    def active(self, owner):
        with self.lock:
            return next((j["id"] for j in reversed(list(self.jobs.values())) if j["owner"]==owner and j["state"] in ("queued","running")),None)
    def get(self, owner, ident):
        with self.lock:
            job = self.jobs.get(ident)
            if not job or job['owner'] != owner:
                return None
            return deepcopy({k:v for k,v in job.items() if k not in ('event','owner')})
    def cancel(self, owner, ident):
        with self.lock:
            job = self.jobs.get(ident)
            if not job or job['owner'] != owner:
                raise JobError('작업을 찾을 수 없습니다.')
            if job['state'] in ('queued','running'):
                job['event'].set()
    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=False)
