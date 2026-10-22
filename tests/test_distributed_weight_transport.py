"""Public full-update transport works without loading actor-side Megatron patches."""
from argparse import Namespace
import sys
import unittest
from unittest.mock import MagicMock,patch

import torch
from slime.utils import weight_sync as sync


class TransportTest(unittest.TestCase):
    def test_import_does_not_load_trainer_patches(self):
        self.assertNotIn('slime.backends.megatron_utils',sys.modules)

    def test_metadata_precedes_broadcast_and_requests_are_returned(self):
        engine=MagicMock();engine.update_weights_from_distributed.remote.return_value='request'
        tensor=torch.ones(2,3)
        with patch.object(sync.dist,'broadcast') as broadcast:
            refs=sync.update_weights_from_distributed('group','pg',16,[engine],[('layer.weight',tensor)])
        self.assertEqual(refs,['request'])
        kwargs=engine.update_weights_from_distributed.remote.call_args.kwargs
        self.assertEqual(kwargs['names'],['layer.weight'])
        self.assertEqual(kwargs['shapes'],[torch.Size([2,3])])
        self.assertEqual(kwargs['weight_version'],'16')
        broadcast.assert_called_once();broadcast.return_value.wait.assert_called_once()

    def test_connect_ranks_and_full_nccl_backend(self):
        engines=[MagicMock(),MagicMock()]
        with patch.object(sync,'init_process_group',return_value='group') as create,patch.object(sync.ray,'get'):
            group=sync.connect_rollout_engines_from_distributed(Namespace(rollout_num_gpus_per_engine=1),
                        'test',engines,engine_gpu_counts=[1,1])
        self.assertEqual(group,'group')
        self.assertEqual(create.call_args.kwargs['backend'],'nccl')
        self.assertEqual(create.call_args.kwargs['world_size'],3)
        for i,engine in enumerate(engines):
            kwargs=engine.init_weights_update_group.remote.call_args.kwargs
            self.assertEqual(kwargs['rank_offset'],i+1)
            self.assertEqual(kwargs['world_size'],3)


if __name__=='__main__':unittest.main()
