import unittest
import numpy as np
from exec_comp.round3.boundary_cache import BoundaryCache
from exec_comp.round3.vendor import activate
activate()
from scalebridge.online.adaptive_chunk_execution import closing_boundary


class BoundaryMemoTests(unittest.TestCase):
    def test_exact_for_every_cursor_and_inplace_tip_event_changes(self):
        original=getattr(closing_boundary,'original',closing_boundary);cache=BoundaryCache(original)
        rng=np.random.default_rng(40)
        for case in range(12):
            chunk={'frame_count':100,'left_fingertip_local':rng.normal(0,.06,(100,5,3))}
            hand={'events':[{'kind':'close','start':10,'end':22},{'kind':'open','start':40,'end':48},
                            {'kind':'close','start':70,'end':88}]} if case%2 else None
            for cursor in np.arange(-1,102,.3):
                expected=original(chunk,'left',cursor,hand)
                self.assertEqual(cache(chunk,'left',cursor,hand),expected)
                self.assertEqual(cache(chunk,'left',cursor,hand),expected)
            chunk['left_fingertip_local'][:]=0.
            if hand:hand['events'][0]['start']=0;hand['events'][0]['end']=99
            for cursor in (0.,10.5,71.,99.):
                self.assertEqual(cache(chunk,'left',cursor,hand),original(chunk,'left',cursor,hand))
        self.assertGreater(cache.hits,cache.misses)
