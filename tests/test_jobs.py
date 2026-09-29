from threading import Event
import pytest
from jobs import JobManager, JobError


def test_owner_isolation_and_rerun_deduplication():
    manager=JobManager(1);entered=Event();release=Event()
    def work(ctx):
        entered.set();release.wait(3);ctx.progress(.5,'actual stage');return {'ok':True}
    job=manager.submit('alice','analysis',work)
    assert entered.wait(2)
    assert manager.get('bob',job) is None
    with pytest.raises(JobError): manager.cancel('bob',job)
    with pytest.raises(JobError): manager.submit('alice','again',work)
    release.set();manager.close()
    result=manager.get('alice',job)
    assert result['state']=='completed' and result['result']=={'ok':True}
    result['result']['ok']=False
    assert manager.get('alice',job)['result']['ok'] is True


def test_failure_and_cooperative_cancel():
    manager=JobManager(1);entered=Event();release=Event()
    def work(ctx):
        entered.set();release.wait(3);ctx.check_cancelled()
    job=manager.submit('alice','analysis',work);assert entered.wait(2)
    manager.cancel('alice',job);release.set();manager.close()
    assert manager.get('alice',job)['state']=='cancelled'
    manager=JobManager(1)
    def broken(ctx): raise ValueError('broken input')
    job=manager.submit('bob','analysis',broken);manager.close()
    assert manager.get('bob',job)['state']=='failed'
    assert 'broken input' in manager.get('bob',job)['error']


def test_active_work_can_be_rediscovered_by_its_owner():
    manager=JobManager(1);entered=Event();release=Event()
    def work(ctx): entered.set();release.wait(3)
    job=manager.submit('alice','analysis',work);assert entered.wait(2)
    assert manager.active('alice')==job
    assert manager.active('bob') is None
    release.set();manager.close()
    assert manager.active('alice') is None
