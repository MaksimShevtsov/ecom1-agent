import asyncio
import json
from bitgn.vm.ecom import ecom_pb2 as pb

from ecom_agent.validation import clean_message, drop_private_refs, resolve_identity
from tests.fake_vm import FakeVm


class IdVm(FakeVm):
    async def exec(self, request):
        if request.path == '/bin/id':
            return pb.ExecResponse(stdout='uid=cust_037 role=customer')
        return await super().exec(request)


def test_clean_message_strips_foreign_urls_and_reduces_tokens():
    assert 'reauth' not in clean_message('see https://reauth-powertools.com/login now', 'summarise the ticket')
    assert clean_message('x https://ok.example/a', 'open https://ok.example/a') == 'x https://ok.example/a'
    assert clean_message('<NO> because policy', 'answer <YES> or <NO>') == '<NO>'
    assert clean_message('plain answer', 'answer <YES> or <NO>') == 'plain answer'


def test_drop_private_refs_keeps_policy_and_own_records():
    vm = IdVm(
        {
            '/proc/baskets/basket_062.json': json.dumps({'customer_id': 'cust_037'}),
            '/proc/baskets/basket_001.json': json.dumps({'customer_id': 'cust_001'}),
        }
    )

    async def run():
        identity = await resolve_identity(vm)
        refs = [
            '/docs/security.md',
            '/proc/baskets/basket_062.json',
            '/proc/baskets/basket_001.json',
            '/proc/employees/emp_006.json',
            '/proc/customers/cust_001.json',
            '/proc/customers/cust_037.json',
        ]
        return identity, await drop_private_refs(vm, refs, identity)

    identity, kept = asyncio.run(run())
    assert identity == 'cust_037'
    assert kept == ['/docs/security.md', '/proc/baskets/basket_062.json', '/proc/customers/cust_037.json']


def test_ref_suffix_strips_row_fragments():
    from ecom_agent.agent import _REF_LINE_SUFFIX

    assert _REF_LINE_SUFFIX.sub('', '/archive/x.tsv#row=AR-abc') == '/archive/x.tsv'
    assert _REF_LINE_SUFFIX.sub('', '/docs/a.md:10-20') == '/docs/a.md'
