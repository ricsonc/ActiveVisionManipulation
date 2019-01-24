from gym.envs.registration import register
import HER

register(
    id='pusher-v0',
    entry_point='HER.envs.pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='pusher-v1',
    entry_point='HER.envs.close_pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='oc_pusher-v0',
    entry_point='HER.envs.oc_pusher:BaxterEnv',
    kwargs={'max_len':20}
)


register(
    id='img_pusher-v0',
    entry_point='HER.envs.oc_pusher:BaxterEnv',
    kwargs={'max_len':20, 'img': True}
)

register(
    id='bb_pusher-v0',
    entry_point='HER.envs.bb_pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='bb_pusher-v1',
    entry_point='HER.envs.bb_pusher:BaxterEnv',
    kwargs={'max_len':20, 'bbox_noise': 2.0}
)

register(
    id='fakercnn_pusher-v0',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'max_len':20, 'bbox_noise': 0.0}
)

register(
    id='img_pusher-v1',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'max_len':20, 'img': True}
)

register(
    id='fakercnn_pusher-v1',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml'}
)

register(
    id='fakercnn_pusher-v2',
    entry_point='HER.envs.fakercnn_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'bbox_noise': 1.5}
)

#nothing to do here... just debugging, static camera
register(
    id='active_pusher-v0',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0}
)

register(
    id='active_pusher-v1',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0, 'rot': True, 'rot_scale': 0.0}
)

register(
    id='active_pusher-v2',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={}
)

register(
    id='active_pusher-v5',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'rot': True}
)


register(
    id='active_pusher-v3',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml'}
)

register(
    id='active_pusher-v30',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'bbox_noise': 0.5}
)

register(
    id='active_pusher-v31',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'fake_scheme': 'piecewise'}
)

register(
    id='active_pusher-v32',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'fake_scheme': 'piecewise', 'bbox_noise': 1}
)

#handicapped distractor
register(
    id='active_pusher-v4',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'pos_scale': 0.0}
)

#with auxiliary rwd
register(
    id='active_pusher-v6',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'aux_rwd': True}
)

register(
    id='active_pusher-v7',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'aux_rwd': True, 'pos_scale': 0.0}
)

register(
    id='active_pusher-v8',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'aux_rwd': False, 'randcam': True}
)

#rcnn trajs
register(
    id='rcnn_pusher-v0',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'test': True}
)
register(
    id='rcnn_pusher-v1',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'pos_scale': 0.0, 'test': True}
)

register(
    id='rcnn_pusher-v2',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'filename': 'mjc/distractor.xml', 'test': True, 'randcam': True}
)

#like active_pusher but img only
register(
    id='imgonly_pusher-v0',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0, 'not_oc': True}
)

register(
    id='imgonly_pusher-v1',
    entry_point='HER.envs.active_pusher:BaxterEnv',
    kwargs={'pos_scale': 0.0, 'not_oc': True, 'no_obj': True}
)

for i, noise_level in enumerate([0, 1, 2]):
    for j, pos_scale in enumerate([0.015, 0.03, 0.06, 0.00]):
        for k, bound in enumerate([0.2, 0.4]):
            kwargss = [{}, {'filename': 'mjc/distractor.xml'}]
            kwargss.append(kwargss[-1])
            kwargss[-1]['aux_rwd'] = True
            
            const_kwargs = {
                'fake_scheme': 'piecewise',
                'bbox_noise': noise_level,
                'pos_scale': pos_scale,
                'bound': bound,
            }

            #active pusher on distractor env
            register(
                id='active_pusher_dist_%d_%d_%d-v0' % (i,j,k),
                entry_point = 'HER.envs.active_pusher:BaxterEnv',
                kwargs=dict(kwargss[1], **const_kwargs),
            )

            #active pusher on distractor env with aux
            register(
                id='active_pusher_dist_aux_%d_%d_%d-v0' % (i,j,k),
                entry_point = 'HER.envs.active_pusher:BaxterEnv',
                kwargs=dict(kwargss[2], **const_kwargs),
            )
            
            #pretraining active pusher w/o distractors
            register(
                id='active_pusher_pret_%d_%d_%d-v0' % (i,j,k),
                entry_point = 'HER.envs.active_pusher:BaxterEnv',
                kwargs=dict(kwargss[0], **const_kwargs),
            )

            #testing envs for the active pushers
            register(
                id='rcnn_pusher_%d_%d_%d-v0' % (i,j,k),
                entry_point = 'HER.envs.active_pusher:BaxterEnv',
                kwargs=dict(kwargss[1], test=True, **const_kwargs),
            )
            
            #testing envs for the active pushers
            register(
                id='rcnn_randpusher_%d_%d_%d-v0' % (i,j,k),
                entry_point = 'HER.envs.active_pusher:BaxterEnv',
                kwargs=dict(kwargss[1], test=True, randcam = True, **const_kwargs),
            )

            #random cam pretraining
            register(
                id='randcam_pusher_pret_%d_%d_%d-v0' % (i,j,k),
                entry_point = 'HER.envs.active_pusher:BaxterEnv',
                kwargs=dict(kwargss[0], randcam = True, **const_kwargs),
            )

            #random cam distractors
            register(
                id='randcam_pusher_dist_%d_%d_%d-v0' % (i,j,k),
                entry_point = 'HER.envs.active_pusher:BaxterEnv',
                kwargs=dict(kwargss[1], randcam = True, **const_kwargs),
            )

